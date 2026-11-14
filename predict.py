"""Prediction Module for Movie Review Sentiment Analysis.

Provides the SentimentPredictor class:
- Loads the trained pipeline artifact once into memory
- Preprocesses raw review inputs with clean_text
- Computes genuine model probability and sentiment label (Positive / Negative)
- Detects borderline / uncertain predictions where the model lacks strong confidence
- Handles empty, malformed, and extremely long reviews safely
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from sklearn.pipeline import Pipeline

from src.text_preprocessing import clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Documentation on Borderline Sentiment:
# The IMDB dataset contains polarized positive and negative reviews.
# It does not train an explicit neutral class.
# When a review's predicted probability falls in the uncertain window [0.42, 0.58],
# we flag it as a borderline prediction to avoid false certainty.
BORDERLINE_LOWER_THRESHOLD = 0.42
BORDERLINE_UPPER_THRESHOLD = 0.58


class SentimentPredictor:
    """Production-grade sentiment predictor for movie reviews."""

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        """Initialize and load the model once into memory.

        Args:
            model_path: Optional path to the trained Joblib pipeline file.
                        If None, searches the default 'models/sentiment_pipeline.joblib'.
        """
        self.pipeline: Optional[Pipeline] = None
        self.model_path: Optional[Path] = None
        self.is_loaded: bool = False

        default_path = Path("models") / "sentiment_pipeline.joblib"
        target_path = Path(model_path) if model_path else default_path

        if target_path.exists():
            self.load_model(target_path)
        else:
            logger.warning("Model file not found at: %s. Predictor initialized unloaded.", target_path)

    def load_model(self, path: Path | str) -> None:
        """Load the trained pipeline artifact from disk.

        Args:
            path: Path to the Joblib artifact file.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If deserialization fails.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file does not exist: {p.resolve()}")

        try:
            logger.info("Loading sentiment pipeline from %s...", p)
            self.pipeline = joblib.load(p)
            self.model_path = p
            self.is_loaded = True
            logger.info("Model loaded successfully.")
        except Exception as exc:
            self.is_loaded = False
            raise RuntimeError(f"Failed to load model artifact: {exc}") from exc

    def predict(self, text: Optional[str]) -> Dict[str, Any]:
        """Analyze an English movie review and predict sentiment.

        Args:
            text: Raw English review text.

        Returns:
            Dictionary containing:
            - success: bool
            - sentiment: 'Positive' or 'Negative' (or None on failure)
            - label_id: 1 for positive, 0 for negative
            - pos_probability: float in [0.0, 1.0]
            - neg_probability: float in [0.0, 1.0]
            - confidence: float in [0.5, 1.0]
            - is_borderline: bool
            - borderline_message: explanatory notice when borderline
            - error: error message string if validation failed
            - character_count: length of original input
            - word_count: approximate word count
        """
        if not self.is_loaded or self.pipeline is None:
            return {
                "success": False,
                "error": "Model is not loaded. Please train or load a valid model first.",
                "sentiment": None,
                "is_borderline": False,
            }

        # Input validation
        if text is None or not isinstance(text, str):
            return {
                "success": False,
                "error": "Review text must be a valid non-empty string.",
                "sentiment": None,
                "is_borderline": False,
            }

        raw_stripped = text.strip()
        if not raw_stripped:
            return {
                "success": False,
                "error": "Review text cannot be empty or whitespace only.",
                "sentiment": None,
                "is_borderline": False,
            }

        # Clean text using identical preprocessing as training
        normalized_text = clean_text(raw_stripped)
        if not normalized_text:
            return {
                "success": False,
                "error": "Review contains no readable words after standard text preprocessing.",
                "sentiment": None,
                "is_borderline": False,
            }

        # Run inference through pipeline
        try:
            # Most pipelines support predict_proba
            if hasattr(self.pipeline, "predict_proba"):
                probs = self.pipeline.predict_proba([normalized_text])[0]
                neg_prob = float(probs[0])
                pos_prob = float(probs[1])
            elif hasattr(self.pipeline, "decision_function"):
                # Fallback for margin-based classifiers
                import numpy as np
                score = float(self.pipeline.decision_function([normalized_text])[0])
                pos_prob = 1.0 / (1.0 + np.exp(-score))
                neg_prob = 1.0 - pos_prob
            else:
                pred_label = int(self.pipeline.predict([normalized_text])[0])
                pos_prob = 1.0 if pred_label == 1 else 0.0
                neg_prob = 1.0 - pos_prob

            pred_label = 1 if pos_prob >= 0.5 else 0
            sentiment_str = "Positive" if pred_label == 1 else "Negative"
            confidence = pos_prob if pred_label == 1 else neg_prob

            # Detect borderline/uncertain sentiment
            is_borderline = BORDERLINE_LOWER_THRESHOLD <= pos_prob <= BORDERLINE_UPPER_THRESHOLD
            borderline_msg = (
                "Borderline sentiment — the model is not confident about this review."
                if is_borderline
                else ""
            )

            word_count = len(normalized_text.split())

            return {
                "success": True,
                "sentiment": sentiment_str,
                "label_id": pred_label,
                "pos_probability": round(pos_prob, 4),
                "neg_probability": round(neg_prob, 4),
                "confidence": round(confidence, 4),
                "is_borderline": is_borderline,
                "borderline_message": borderline_msg,
                "character_count": len(raw_stripped),
                "word_count": word_count,
                "error": None,
            }

        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return {
                "success": False,
                "error": f"Inference execution failed: {str(exc)}",
                "sentiment": None,
                "is_borderline": False,
            }


# Singleton instance for high-efficiency application reuse
_global_predictor: Optional[SentimentPredictor] = None


def get_predictor(model_path: Optional[Path | str] = None) -> SentimentPredictor:
    """Return the global cached SentimentPredictor instance to avoid reloading overhead."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = SentimentPredictor(model_path)
    elif model_path and _global_predictor.model_path != Path(model_path):
        _global_predictor.load_model(model_path)
    return _global_predictor


if __name__ == "__main__":
    import sys

    sample_text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "This masterpiece had breathtaking performances and brilliant cinematography!"
    )

    predictor = get_predictor()
    if not predictor.is_loaded:
        print("ERROR: Model is not loaded. Train the model first via train.py.")
        sys.exit(1)

    result = predictor.predict(sample_text)
    print("\n--- Sentiment Analysis Result ---")
    print(f"Input: {sample_text}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Positive Probability: {result['pos_probability']:.2%}")
    print(f"Negative Probability: {result['neg_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
    if result["is_borderline"]:
        print(f"Notice: {result['borderline_message']}")

