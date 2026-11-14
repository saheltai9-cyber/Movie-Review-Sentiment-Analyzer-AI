# Movie Review Intelligence Studio (Project 04 Upgrade)
### Stanford IMDb Sentiment Analysis AI & Public Movie Discovery Studio

---

## 1. Project Overview

**Movie Review Intelligence Studio** is a full-featured Machine Learning web application designed to discover movies online from public sources, showcase rich movie synopsis and posters, and analyze user reviews using a local, classical Machine Learning pipeline trained on the **Stanford Large Movie Review Dataset** (50,000 reviews).

Key capabilities:
- **Zero API Setup**: Public movie search via open Wikipedia and MediaWiki endpoints without requiring API keys, tokens, developer accounts, or paid services.
- **Contextual Movie Association**: Associates reviews with the selected movie while keeping sentiment scoring 100% focused on the user's review text.
- **Manual Fallback**: Provides a manual movie entry modal if a title cannot be found online or when the system is offline.
- **Negation-Preserving Classical ML**: Sublinear TF-IDF ($1 + \log(\text{tf})$) unigram/bigram vectorizer and L2-regularized Logistic Regression yielding **88.97% test accuracy** and **0.9577 ROC-AUC** on 25,000 official test reviews.
- **Transparent Borderline Uncertainty**: Automatically detects and flags reviews with ambiguous probabilities ($0.42 \le P(\text{Positive}) \le 0.58$).
- **Strict WCAG AA Contrast**: Fully custom theme engine supporting both Dark Mode (`#0B1220`) and Light Mode (`#F6F8FC`) with crystal-clear readability and contrast.
- **Complete Review Privacy**: Reviews are evaluated locally on the CPU; review text is never transmitted to external APIs or services.

---

## 2. Key Workflow

```
1. Movie Search Tab
   │
   ├─► User enters movie title & optional year
   ├─► Real-time query to Wikipedia public endpoint (no credentials needed)
   ├─► User inspects real results (Title, Year, Genres, Synopsis, Poster)
   └─► User selects movie OR enters details manually
           │
           ▼
2. Sentiment Analyzer Tab
   │
   ├─► Displays Selected Movie Banner (poster, year, genres, overview)
   ├─► User enters or pastes review text for that movie
   ├─► User clicks 'Analyze Sentiment'
   └─► Local ML Pipeline executes (< 10 ms inference)
           │
           ▼
3. Prediction Results
   │
   ├─► Associated Movie Title
   ├─► Positive / Negative / Borderline Badge
   ├─► Dual-Tone Probability Meter (Positive % vs. Negative %)
   ├─► Borderline Warning Callout (if 0.42 <= P <= 0.58)
   └─► Workflow Actions: 'Analyze Another Review' or 'Change Movie'
```

---

## 3. Public Data Sources & Attribution

- **Movie Metadata & Imagery**: Retrieved dynamically from the **Wikipedia Action API** (`en.wikipedia.org/w/api.php`) and **Wikipedia REST API** (`en.wikipedia.org/api/rest_v1/page/summary/`) under Creative Commons Attribution-ShareAlike licenses (CC BY-SA).
- **Zero Credentials**: Uses publicly available, unauthenticated HTTP GET requests. Requires no TMDB API keys, accounts, or developer tokens.
- **Stanford IMDb Benchmark**: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). *Learning Word Vectors for Sentiment Analysis*. Association for Computational Linguistics (ACL).
  - 50,000 reviews ($25,000$ train, $25,000$ test; $12,500$ pos & $12,500$ neg in each).

---

## 4. Machine Learning Metrics & Architecture

The model was selected via an 80/20 stratified validation comparison between Logistic Regression and Multinomial Naive Bayes, then refit on 100% of the training set ($25,000$ reviews) and evaluated on the untouched test set ($25,000$ reviews).

| Metric | Real Test-Set Value | Description |
|---|---|---|
| **Accuracy** | **0.8897 (88.97%)** | Correct on 22,242 / 25,000 test reviews |
| **Precision** | **0.8862 (88.62%)** | Positive predictive accuracy |
| **Recall** | **0.8942 (89.42%)** | Sensitivity to positive sentiment |
| **F1 Score** | **0.8902 (89.02%)** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9577** | Area under the ROC curve |
| **Inference Time** | **< 10 ms** | Real-time CPU inference per review |

### Test-Set Confusion Matrix ($N = 25,000$)
```
                         PREDICTED NEGATIVE    PREDICTED POSITIVE
ACTUAL NEGATIVE (12,500)       11,064 (TN)           1,436 (FP)
ACTUAL POSITIVE (12,500)        1,322 (FN)          11,178 (TP)
```

---

## 5. Directory Structure

```text
04-Movie-Review-Sentiment-Analyzer-AI/
├── app.py                      # Upgraded NiceGUI studio web application
├── train.py                    # Model training entrypoint (stratified comparison & refit)
├── predict.py                  # SentimentPredictor inference engine
├── download_data.py            # Stanford dataset downloader with multi-mirror fallback
├── package_project.py          # Portable ZIP packager and clean-room test runner
├── requirements.txt            # Python package dependencies
├── README.md                   # System documentation
├── run.bat                     # Windows one-click launcher
├── .gitignore                  # Git exclusions
│
├── src/
│   ├── __init__.py
│   ├── movie_search.py         # Public Wikipedia movie discovery & metadata service
│   ├── data_loader.py          # Multithreaded reader & folder structure validator
│   ├── text_preprocessing.py   # HTML cleaning & negation-safe normalizer
│   ├── model_training.py       # TF-IDF pipeline builder & model selection logic
│   ├── evaluation.py           # Metric calculation (Acc, F1, AUC, Confusion Matrix)
│   └── ui_components.py        # Reusable movie cards, meters, badges, and grids
│
├── data/
│   ├── raw/aclImdb/            # Official Stanford dataset (50,000 .txt reviews)
│   └── processed/              # High-speed disk cache for instant re-loads
│
├── models/
│   ├── sentiment_pipeline.joblib # Serialized trained scikit-learn pipeline (485 KB)
│   └── model_metadata.json     # Architecture specs and model parameters
│
├── reports/
│   ├── model_metrics.json      # Genuine computed test-set metrics
│   ├── evaluation_report.md    # Formatted evaluation report
│   └── FINAL_RELEASE_REPORT.md # Comprehensive engineering & release audit
│
├── tests/
│   ├── test_movie_search.py    # Public search, mock parsing, offline fallback tests
│   ├── test_preprocessing.py   # HTML tags, entities, negations, contractions
│   ├── test_data_loader.py     # Folder validation & label mapping tests
│   ├── test_model_pipeline.py  # Pipeline fit, predict, and serialization tests
│   └── test_prediction.py      # Empty inputs, thresholds, and live inference tests
│
└── dist/
    └── Movie-Review-Sentiment-Analyzer-AI.zip # Verified portable standalone package
```

---

## 6. Installation & Quick Start

### Quick Launch on Windows
Double-click [`run.bat`](file:///c:/Users/sahil/OneDrive/Desktop/PROJECT%20NUMBER%204/04-Movie-Review-Sentiment-Analyzer-AI/run.bat) or run in terminal:

```cmd
run.bat
```

The launcher:
1. Detects Python 3.11/3.12.
2. Creates and activates the local `.venv`.
3. Installs dependencies from `requirements.txt`.
4. Checks for the existing trained model artifact (`models/sentiment_pipeline.joblib`).
5. Starts the NiceGUI server at `http://127.0.0.1:8080`.

### Manual Launch
```bash
cd 04-Movie-Review-Sentiment-Analyzer-AI
python app.py --host 127.0.0.1 --port 8080
```
Open your browser at `http://127.0.0.1:8080`.

---

## 7. Running the Test Suite

Execute the 23 unit tests via `pytest`:

```bash
pytest tests -v
```

All 23 tests cover:
- Online & offline movie discovery, caching, and fallback.
- Consistency test: submitting identical review text across different selected movies produces identical sentiment scores.
- Negation preservation (`not`, `no`, `never`, `barely`, `hardly`) in stop words.
- HTML tag stripping, whitespace normalization, and long input truncation.
- Pipeline training, prediction consistency, and serialization round-trip.
- Probability distribution constraints and borderline detection thresholds.

---

## 8. Operational Boundaries & Honest Disclosures

1. **Polarity-Only Training**: The IMDB benchmark explicitly excludes neutral reviews (ratings 5-6). Therefore, the model **does not support an authentic 3-class neutral category**. Uncertain reviews are flagged with a *Borderline sentiment* notice.
2. **Contextual Independence**: The model analyzes only the text of the review. The selected movie title and synopsis do not bias or distort the sentiment prediction.
3. **Domain Focus**: Optimized for film critique language.
4. **User Privacy**: Review text is analyzed strictly on the local machine and is never sent to external servers.
