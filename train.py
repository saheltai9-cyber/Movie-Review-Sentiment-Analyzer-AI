"""Model Training Entrypoint for Movie Review Sentiment Analyzer AI.

Orchestrates:
1. Dataset verification and loading from data/raw/aclImdb
2. Stratified 80/20 train/validation split
3. Model comparison: Logistic Regression vs Multinomial Naive Bayes
4. Selection of best model via validation F1 score
5. Final refitting on complete 25,000 review training set
6. Unbiased evaluation on 25,000 review official test set
7. Saving trained pipeline artifact and genuine evaluation metrics
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from download_data import ensure_imdb_dataset
from src.data_loader import load_imdb_dataset, validate_dataset_structure
from src.model_training import train_and_evaluate_final_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrainPipeline")


def run_training_pipeline(
    data_dir: str = "data/raw/aclImdb",
    models_dir: str = "models",
    reports_dir: str = "reports",
    sample_limit: int | None = None,
    random_state: int = 42,
) -> dict:
    """Execute the end-to-end sentiment analyzer training pipeline.

    Args:
        data_dir: Path to aclImdb directory.
        models_dir: Destination folder for trained model artifacts.
        reports_dir: Destination folder for metrics reports.
        sample_limit: Optional sample cap per class for testing/debugging.
        random_state: Reproducibility random seed.

    Returns:
        Dictionary of final test evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info("MOVIE REVIEW SENTIMENT ANALYZER AI - TRAINING PIPELINE")
    logger.info("=" * 60)

    dataset_path = Path(data_dir)

    # Verify or acquire dataset
    val = validate_dataset_structure(dataset_path)
    if not val["valid"]:
        logger.warning(
            "Dataset not found or incomplete at %s. Attempting auto-retrieval...",
            dataset_path,
        )
        try:
            parent_dir = dataset_path.parent if dataset_path.name == "aclImdb" else dataset_path
            dataset_path = ensure_imdb_dataset(parent_dir)
        except Exception as exc:
            logger.error("Dataset preparation failed: %s", exc)
            raise RuntimeError(
                f"Cannot proceed without valid Stanford IMDB dataset: {exc}"
            ) from exc

    # Load dataset
    logger.info("Loading reviews from %s...", dataset_path)
    start_load = time.time()
    X_train, y_train, X_test, y_test = load_imdb_dataset(
        dataset_path, limit_per_class=sample_limit
    )
    logger.info("Dataset loaded in %.2f seconds.", time.time() - start_load)

    # Train, compare, refit, and evaluate
    metrics = train_and_evaluate_final_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models_dir=models_dir,
        reports_dir=reports_dir,
        random_state=random_state,
    )

    # Print human-readable summary
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION METRICS ON OFFICIAL TEST SET:")
    logger.info("  Selected Model: %s", metrics.get("model_name"))
    logger.info("  Accuracy:       %.4f (%.2f%%)", metrics["accuracy"], metrics["accuracy"] * 100)
    logger.info("  Precision:      %.4f", metrics["precision"])
    logger.info("  Recall:         %.4f", metrics["recall"])
    logger.info("  F1 Score:       %.4f", metrics["f1_score"])
    logger.info("  ROC-AUC:        %s", str(metrics.get("roc_auc")))
    cm = metrics["confusion_matrix"]
    logger.info(
        "  Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d",
        cm["tn"],
        cm["fp"],
        cm["fn"],
        cm["tp"],
    )
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentiment Analyzer Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/aclImdb",
        help="Path to aclImdb directory (default: data/raw/aclImdb)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Folder to save trained pipeline (default: models)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Folder to save metrics reports (default: reports)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional sample limit per class for fast testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    try:
        run_training_pipeline(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            sample_limit=args.sample_limit,
            random_state=args.random_state,
        )
        print("\nTRAINING PIPELINE COMPLETED SUCCESSFULLY.")
    except Exception as e:
        logger.exception("Training failed: %s", e)
        sys.exit(1)

