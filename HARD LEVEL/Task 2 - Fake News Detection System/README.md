# Fake News Detection System

## Overview

The Fake News Detection System is a local, classical Natural Language Processing (NLP) and machine learning application developed for the Kodbud Solutions internship (Hard Level Task 2). The system processes full-text news articles and headlines, extracts statistical text features using Term Frequency-Inverse Document Frequency (TF-IDF), and classifies articles as either `REAL` or `FAKE` with a calibrated confidence score.

The project runs entirely offline on standard desktop hardware using open-source Python libraries without relying on external APIs, cloud inference, or paid services.

## Objective

To build a reproducible, leakage-free machine learning classification system capable of distinguishing between real and fabricated news articles using supervised text classification, evaluating multiple models on a held-out test set, persisting the best-performing pipeline for fast offline inference, and providing both command-line and local web interfaces.

## Features

- End-to-end data validation, normalization, text cleaning, and deduplication.
- Strict prevention of data leakage by splitting the dataset prior to fitting the TF-IDF vectorizer.
- Automated training and comparison of three supervised classifiers:
  - Logistic Regression
  - Calibrated Linear Support Vector Classifier (`LinearSVC` with sigmoid calibration)
  - Multinomial Naive Bayes
- Comprehensive evaluation metrics on a held-out 20% test split (Accuracy, Precision, Recall, F1-score, Classification Report, and Confusion Matrix).
- Automated generation of visual evaluation artifacts (`confusion_matrix.png`, `class_distribution.png`, `model_comparison.csv`, and `evaluation_report.txt`).
- Model and vectorizer persistence using `joblib` (`models/fake_news_model.pkl` and `models/tfidf_vectorizer.pkl`).
- Interactive command-line interface (`fake_news_detector.py`) and local Streamlit web application (`app.py`).

## Technology Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- joblib
- Streamlit

## Dataset

The project uses the open-source Fake and Real News benchmark dataset (`news.csv` / `fake_or_real_news.csv`), and also supports the two-file ISOT Fake and Real News dataset (`Fake.csv` and `True.csv`).

- Total raw records in `news.csv`: 6,335 articles
- Validated and deduplicated records: 6,270 unique articles
  - `FAKE`: 3,116 articles (49.70%)
  - `REAL`: 3,154 articles (50.30%)
- Core fields supported: `title`, `text`, `subject`, `date`, `label`

## Dataset Preparation

Because raw news datasets are large (~30.7 MB for `news.csv`), raw CSV files are excluded from Git tracking via `.gitignore`. Detailed download instructions are provided in `dataset/README.md`.

To place the dataset in the project directory on Windows PowerShell:

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv" -OutFile "dataset/news.csv"
```

Alternatively, place `Fake.csv` and `True.csv` from the ISOT dataset directly inside the `dataset/` directory. If the dataset is missing, `fake_news_detector.py` reports a descriptive error message indicating where to download and place the files.

## Machine Learning Workflow

```text
News Dataset
      |
      v
Data Cleaning & Deduplication
      |
      v
Stratified Train/Test Split (80% Train / 20% Test)
      |
      v
TF-IDF Vectorization (Fit on Train Only)
      |
      v
Machine Learning Classification (Logistic Regression, LinearSVC, MultinomialNB)
      |
      v
Model Evaluation on Held-Out Test Set
      |
      v
Save Best Model & Vectorizer (joblib)
      |
      v
User Enters Article -> Prediction (REAL / FAKE + Confidence)
```

## Data Preprocessing

The preprocessing pipeline in `clean_text()` and `prepare_data()` performs the following operations:

1. Standardizes heterogeneous column names (`title`, `text`, `label`) and normalizes label values into `FAKE` (`0`) and `REAL` (`1`).
2. Removes records with missing labels or empty article bodies.
3. Combines the article headline (`title`) and body (`text`) into a unified document string.
4. Strips HTML tags, URLs, and wire-service dateline prefixes (such as `(Reuters) -`) so the classifier learns content patterns rather than publisher headers.
5. Converts text to lowercase and normalizes whitespace without aggressively stripping meaningful tokens.
6. Removes duplicate articles (29 duplicate records removed from the raw dataset).

## TF-IDF Feature Extraction

To prevent data leakage, the dataset is split into training (5,016 articles) and testing (1,254 articles) sets using stratified sampling (`random_state=42`) before feature extraction.

`TfidfVectorizer` is fitted strictly on `X_train` and applied to `X_test` using `vectorizer.transform(X_test)`.

Final TF-IDF configuration:

- `ngram_range=(1, 2)` (unigrams and bigrams)
- `min_df=3` (ignores rare terms appearing in fewer than 3 training documents)
- `max_df=0.90` (ignores terms appearing in more than 90% of training documents)
- `sublinear_tf=True` (applies logarithmic term frequency scaling $1 + \log(\text{tf})$)
- `stop_words="english"`
- `strip_accents="unicode"`
- Resulting vocabulary size on training split: 122,524 features

## Model Training

Three supervised classifiers are trained on the identical training split (`X_train_tfidf`, `y_train`):

1. **Logistic Regression**: `LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)`
2. **LinearSVC (Calibrated)**: `LinearSVC(C=1.0, class_weight="balanced", random_state=42)` wrapped in `CalibratedClassifierCV(cv=5, method="sigmoid")` trained strictly on the training split so that decision margins are converted into calibrated probability estimates via Platt scaling.
3. **Multinomial Naive Bayes**: `MultinomialNB(alpha=0.5)`

## Model Evaluation

All trained models are evaluated on the held-out test set (1,254 articles). The pipeline computes Accuracy, weighted Precision, weighted Recall, weighted F1-score, a per-class Classification Report, and a Confusion Matrix. Evaluation artifacts are saved to the `results/` directory:

- `results/evaluation_report.txt`
- `results/evaluation_metrics.json`
- `results/model_comparison.csv`
- `results/confusion_matrix.png`
- `results/class_distribution.png`

## Prediction

The best-performing model and fitted TF-IDF vectorizer are saved in `models/fake_news_model.pkl` and `models/tfidf_vectorizer.pkl`. During inference, user input is validated, cleaned using `clean_text()`, transformed via `vectorizer.transform()`, and passed to `model.predict()` and `model.predict_proba()` to return the predicted label (`REAL` or `FAKE`) and calibrated probability confidence without retraining or requiring internet access.

Example output:

```text
Enter news article:

The bipartisan Senate committee approved the annual federal infrastructure budget on Thursday following weeks of negotiations between lawmakers and the Treasury Department.

Prediction: REAL
Confidence: 94.95%

This prediction is based on patterns learned from the training dataset.
It does not independently verify the factual accuracy of the news article.
```

## Project Structure

```text
HARD LEVEL/
└── Task 2 - Fake News Detection System/
    ├── fake_news_detector.py
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── dataset/
    │   └── README.md
    ├── models/
    │   ├── .gitkeep
    │   ├── fake_news_model.pkl
    │   └── tfidf_vectorizer.pkl
    └── results/
        ├── .gitkeep
        ├── class_distribution.png
        ├── confusion_matrix.png
        ├── evaluation_metrics.json
        ├── evaluation_report.txt
        └── model_comparison.csv
```

## Installation

Open Windows PowerShell, navigate to the project directory, and run the following commands:

1. Create a virtual environment:

```powershell
python -m venv venv
```

2. Activate the virtual environment:

```powershell
venv\Scripts\Activate.ps1
```

3. Install the required dependencies:

```powershell
pip install -r requirements.txt
```

4. Place the dataset inside `dataset/`:

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv" -OutFile "dataset/news.csv"
```

## Usage

### 1. Train Models and Run Interactive Console

```powershell
python fake_news_detector.py
```

### 2. Classify a Single Article from Command Line Without Retraining

```powershell
python fake_news_detector.py --predict "The Senate passed the bipartisan infrastructure bill after committee review."
```

### 3. Run the Local Streamlit Web Interface

```powershell
streamlit run app.py
```

## Deployment

Platform:
Streamlit Community Cloud (current free tier offering)

Cost:
Free

Inference:
Local scikit-learn model

External AI APIs:
None

### Streamlit Community Cloud Configuration

- Repository: `sohail12345621/Kodbud-Solution-internship`
- Branch: `main`
- Main file path: `HARD LEVEL/Task 2 - Fake News Detection System/app.py`

## Results

All models were evaluated on the held-out test set of 1,254 articles (623 `FAKE`, 631 `REAL`).

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LinearSVC (Calibrated)** | **95.22%** | **95.23%** | **95.22%** | **95.22%** | **Yes** |
| Logistic Regression | 93.46% | 93.55% | 93.46% | 93.46% | No |
| Multinomial Naive Bayes | 92.19% | 92.50% | 92.19% | 92.17% | No |

### Best Model Per-Class Metrics (`LinearSVC (Calibrated)`)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `FAKE` | 0.9433 | 0.9615 | 0.9523 | 623 |
| `REAL` | 0.9612 | 0.9429 | 0.9520 | 631 |
| **Weighted Avg** | **0.9523** | **0.9522** | **0.9522** | **1254** |

### Confusion Matrix (`LinearSVC (Calibrated)`)

| True \ Predicted | Predicted `FAKE` | Predicted `REAL` |
| :--- | :---: | :---: |
| **True `FAKE`** | 599 | 24 |
| **True `REAL`** | 36 | 595 |

## Limitations

- This prediction is based on patterns learned from the training dataset. It does not independently verify the factual accuracy of the news article.
- Because the model relies on bag-of-n-grams TF-IDF representations, it captures lexical and stylistic patterns rather than performing real-time knowledge-graph verification against live events.
- Articles written in domains or time periods substantially different from the training corpus may experience domain shift.

## Future Improvements

- Incorporate cross-domain training datasets covering science, health, financial, and international news.
- Add character-level n-gram features alongside word n-grams to increase robustness against obfuscated spelling.
- Provide feature-importance explanations (top contributing n-grams) for individual article predictions in the user interface.
