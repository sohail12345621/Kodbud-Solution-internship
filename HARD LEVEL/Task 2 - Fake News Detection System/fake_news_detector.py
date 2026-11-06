"""
Fake News Detection System
Hard Level Task 2 - Kodbud Solutions Internship

A local, classical NLP and machine learning pipeline for classifying news
articles as REAL or FAKE using TF-IDF vectorization and supervised classifiers.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure Matplotlib uses a writable temporary directory and non-interactive backend
if "MPLCONFIGDIR" not in os.environ:
    import tempfile
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_PATH = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METRICS_JSON_PATH = RESULTS_DIR / "evaluation_metrics.json"

DISCLAIMER_TEXT = (
    "This prediction is based on patterns learned from the training dataset.\n"
    "It does not independently verify the factual accuracy of the news article."
)

# Numeric label mapping: 0 = FAKE, 1 = REAL
LABEL_TO_ID = {"FAKE": 0, "REAL": 1}
ID_TO_LABEL = {0: "FAKE", 1: "REAL"}


def normalize_label(value: Any) -> Optional[str]:
    """Normalize heterogeneous dataset labels into 'FAKE' or 'REAL'."""
    if pd.isna(value):
        return None
    val_str = str(value).strip().upper()
    if val_str in {"FAKE", "FALSE", "0", "0.0", "UNRELIABLE", "FABRICATED"}:
        return "FAKE"
    if val_str in {"REAL", "TRUE", "1", "1.0", "RELIABLE", "GENUINE"}:
        return "REAL"
    return None


def load_dataset(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """
    Load the fake news dataset from dataset_dir.

    Supports:
    1. Two-file ISOT format: Fake.csv and True.csv (columns: title, text, subject, date)
    2. Single-file format: news.csv, fake_or_real_news.csv, or any CSV with text and label.

    Raises FileNotFoundError with clear instructions if the dataset is missing.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at '{dataset_dir}'.\n"
            "Please create the 'dataset/' folder and place 'news.csv' (or 'Fake.csv' and 'True.csv') inside it.\n"
            "See 'dataset/README.md' for download instructions."
        )

    fake_csv = dataset_dir / "Fake.csv"
    true_csv = dataset_dir / "True.csv"

    if fake_csv.exists() and true_csv.exists():
        print(f"[INFO] Loading ISOT dataset files: {fake_csv.name} and {true_csv.name}")
        df_fake = pd.read_csv(fake_csv, low_memory=False)
        df_true = pd.read_csv(true_csv, low_memory=False)
        df_fake["label"] = "FAKE"
        df_true["label"] = "REAL"
        df = pd.concat([df_fake, df_true], ignore_index=True)
        return df

    candidate_names = [
        "news.csv",
        "fake_or_real_news.csv",
        "fake_news.csv",
        "train.csv",
        "data.csv",
    ]
    selected_file: Optional[Path] = None
    for name in candidate_names:
        candidate = dataset_dir / name
        if candidate.exists():
            selected_file = candidate
            break

    if selected_file is None:
        csv_files = sorted(dataset_dir.glob("*.csv"))
        if csv_files:
            selected_file = csv_files[0]

    if selected_file is None:
        raise FileNotFoundError(
            f"No dataset CSV file found in '{dataset_dir}'.\n"
            "Expected 'dataset/news.csv' (or 'dataset/Fake.csv' and 'dataset/True.csv').\n"
            "Please refer to 'dataset/README.md' for download and placement instructions."
        )

    print(f"[INFO] Loading dataset from: {selected_file}")
    df = pd.read_csv(selected_file, low_memory=False)
    return df


def clean_text(text: Any) -> str:
    """
    Perform reproducible text preprocessing without stripping useful linguistic signal.
    - Converts text to lowercase
    - Removes URLs and HTML tags
    - Removes Reuters/agency dateline prefixes when present to prevent source leakage
    - Collapses redundant whitespace
    """
    if text is None or pd.isna(text):
        return ""
    text_str = str(text)

    # Remove HTML tags and URLs
    text_str = re.sub(r"<[^>]+>", " ", text_str)
    text_str = re.sub(r"https?://\S+|www\.\S+", " ", text_str)

    # Strip agency datelines like "WASHINGTON (Reuters) - " at the start of articles
    text_str = re.sub(
        r"^[A-Z\s,/()]+\((?:Reuters|AP|AFP)\)\s*[-—]+\s*",
        "",
        text_str,
        flags=re.IGNORECASE,
    )

    # Lowercase conversion
    text_str = text_str.lower()

    # Keep alphanumeric characters and basic punctuation spacing clean
    text_str = re.sub(r"[^a-z0-9\s'$%-]", " ", text_str)

    # Collapse multiple whitespace characters into single spaces
    text_str = re.sub(r"\s+", " ", text_str).strip()
    return text_str


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate, clean, deduplicate, and prepare the dataset for modeling.

    Handles:
    - Column standardization (title, text, subject, date, label)
    - Missing values and empty articles
    - Inconsistent labels
    - Duplicate records
    - Title + article body combination
    """
    if df is None or df.empty:
        raise ValueError("The loaded dataset is empty.")

    df = df.copy()

    # Map alternative column names if needed
    col_map = {c.lower().strip(): c for c in df.columns}

    text_col = None
    for candidate in ["text", "content", "article", "body"]:
        if candidate in col_map:
            text_col = col_map[candidate]
            break

    label_col = None
    for candidate in ["label", "class", "target", "category"]:
        if candidate in col_map:
            label_col = col_map[candidate]
            break

    if text_col is None or label_col is None:
        raise ValueError(
            f"Dataset must contain text and label columns. Found columns: {list(df.columns)}"
        )

    title_col = col_map.get("title") or col_map.get("headline")

    initial_rows = len(df)

    # Standardize core columns
    df["raw_text"] = df[text_col]
    df["raw_title"] = df[title_col] if title_col else ""
    df["clean_label"] = df[label_col].apply(normalize_label)

    # Drop rows with missing or unrecognized labels
    df = df.dropna(subset=["clean_label"]).copy()
    after_label_filter = len(df)

    # Fill missing title/text with empty strings and clean both
    df["raw_title"] = df["raw_title"].fillna("").astype(str).str.strip()
    df["raw_text"] = df["raw_text"].fillna("").astype(str).str.strip()

    # Drop records where the article body itself is empty or whitespace-only
    df = df[df["raw_text"].str.len() > 0].copy()

    # Combine title + article text when title is available
    combined_raw = (df["raw_title"] + " " + df["raw_text"]).str.strip()
    df["combined_text"] = combined_raw.apply(clean_text)

    # Remove empty or trivially short articles (fewer than 15 characters after cleaning)
    df = df[df["combined_text"].str.len() >= 15].copy()
    after_empty_filter = len(df)

    # Remove exact duplicate articles
    df = df.drop_duplicates(subset=["combined_text"]).reset_index(drop=True)
    final_rows = len(df)

    df["target"] = df["clean_label"].map(LABEL_TO_ID)

    # Ensure both classes are present
    unique_classes = set(df["clean_label"].unique())
    if unique_classes != {"FAKE", "REAL"}:
        raise ValueError(
            f"Dataset must contain both 'FAKE' and 'REAL' examples. Found: {unique_classes}"
        )

    stats = {
        "initial_rows": int(initial_rows),
        "valid_label_rows": int(after_label_filter),
        "non_empty_rows": int(after_empty_filter),
        "deduplicated_rows": int(final_rows),
        "duplicates_removed": int(after_empty_filter - final_rows),
        "fake_count": int((df["clean_label"] == "FAKE").sum()),
        "real_count": int((df["clean_label"] == "REAL").sum()),
    }
    return df, stats


def build_vectorizer() -> TfidfVectorizer:
    """
    Create the TF-IDF vectorizer with documented parameters.
    Configured with unigrams and bigrams, sublinear TF scaling, and document frequency bounds.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        stop_words="english",
        strip_accents="unicode",
    )


def evaluate_model(
    model: Any,
    X_test_tfidf: Any,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluate a trained classifier on the held-out TF-IDF test features.
    Returns Accuracy, Precision, Recall, F1-score, Classification Report, and Confusion Matrix.
    """
    y_pred = model.predict(X_test_tfidf)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    target_names = [ID_TO_LABEL[0], ID_TO_LABEL[1]]
    report_str = classification_report(
        y_test, y_pred, target_names=target_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "classification_report": report_str,
        "confusion_matrix": cm.tolist(),
    }


def train_models(
    X_train_tfidf: Any,
    y_train: np.ndarray,
    X_test_tfidf: Any,
    y_test: np.ndarray,
) -> Tuple[Any, str, Dict[str, Dict[str, Any]]]:
    """
    Train and compare multiple lightweight machine learning classifiers on the same split:
    1. Logistic Regression
    2. LinearSVC (calibrated via CalibratedClassifierCV on training data for probability output)
    3. Multinomial Naive Bayes

    Selects the best model based on held-out test F1-score (and Accuracy as tie-breaker).
    """
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "LinearSVC (Calibrated)": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=42,
            ),
            cv=5,
            method="sigmoid",
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    }

    evaluations: Dict[str, Dict[str, Any]] = {}
    trained_estimators: Dict[str, Any] = {}

    for name, clf in candidate_models.items():
        print(f"[INFO] Training {name}...")
        clf.fit(X_train_tfidf, y_train)
        trained_estimators[name] = clf
        metrics = evaluate_model(clf, X_test_tfidf, y_test, model_name=name)
        evaluations[name] = metrics
        print(
            f"       -> Accuracy: {metrics['accuracy'] * 100:.2f}% | "
            f"Precision: {metrics['precision'] * 100:.2f}% | "
            f"Recall: {metrics['recall'] * 100:.2f}% | "
            f"F1-Score: {metrics['f1_score'] * 100:.2f}%"
        )

    best_name = max(
        evaluations.keys(),
        key=lambda k: (evaluations[k]["f1_score"], evaluations[k]["accuracy"]),
    )
    best_model = trained_estimators[best_name]
    print(f"\n[INFO] Selected Best Model: {best_name}")
    return best_model, best_name, evaluations


def save_evaluation_artifacts(
    df: pd.DataFrame,
    prep_stats: Dict[str, int],
    split_stats: Dict[str, int],
    best_name: str,
    evaluations: Dict[str, Dict[str, Any]],
    results_dir: Path = RESULTS_DIR,
) -> None:
    """
    Save evaluation metrics, comparison CSV, text report, class distribution chart,
    and confusion matrix visualization into results/.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model comparison table as CSV
    comparison_rows = []
    for name, m in evaluations.items():
        comparison_rows.append(
            {
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1_Score": round(m["f1_score"], 4),
                "Selected": name == best_name,
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)

    # 2. Save confusion matrix plot for the best model
    best_eval = evaluations[best_name]
    cm = np.array(best_eval["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["FAKE", "REAL"],
        yticklabels=["FAKE", "REAL"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 3. Save class distribution plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["clean_label"].value_counts().reindex(["FAKE", "REAL"])
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=["#d9534f", "#5cb85c"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Dataset Class Distribution (Cleaned & Deduplicated)")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Number of Articles")
    for idx, val in enumerate(counts.values):
        ax.text(idx, val + max(counts.values) * 0.01, str(val), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(results_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Save structured JSON metrics for Streamlit / programmatic inspection
    summary_payload = {
        "best_model": best_name,
        "preprocessing_stats": prep_stats,
        "split_stats": split_stats,
        "best_metrics": {
            "accuracy": round(best_eval["accuracy"], 4),
            "precision": round(best_eval["precision"], 4),
            "recall": round(best_eval["recall"], 4),
            "f1_score": round(best_eval["f1_score"], 4),
            "confusion_matrix": best_eval["confusion_matrix"],
        },
        "all_models": {
            name: {
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1_score": round(m["f1_score"], 4),
            }
            for name, m in evaluations.items()
        },
    }
    with open(results_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # 5. Save human-readable evaluation report
    report_lines = [
        "============================================================",
        "FAKE NEWS DETECTION SYSTEM - EVALUATION REPORT",
        "============================================================",
        "",
        "1. DATASET & PREPROCESSING SUMMARY",
        f"   Initial Records:       {prep_stats['initial_rows']}",
        f"   Valid Label Records:   {prep_stats['valid_label_rows']}",
        f"   Non-Empty Records:     {prep_stats['non_empty_rows']}",
        f"   Deduplicated Records:  {prep_stats['deduplicated_rows']} (Removed {prep_stats['duplicates_removed']} duplicates)",
        f"   FAKE Articles:         {prep_stats['fake_count']}",
        f"   REAL Articles:         {prep_stats['real_count']}",
        f"   Training Set Size:     {split_stats['train_size']} (80%)",
        f"   Testing Set Size:      {split_stats['test_size']} (20%)",
        "",
        "2. MODEL COMPARISON (HELD-OUT TEST SET)",
        comparison_df.to_string(index=False),
        "",
        f"3. BEST MODEL DETAILED EVALUATION: {best_name}",
        f"   Accuracy:  {best_eval['accuracy'] * 100:.2f}%",
        f"   Precision: {best_eval['precision'] * 100:.2f}%",
        f"   Recall:    {best_eval['recall'] * 100:.2f}%",
        f"   F1-Score:  {best_eval['f1_score'] * 100:.2f}%",
        "",
        "Classification Report:",
        best_eval["classification_report"],
        "Confusion Matrix (Rows: True [FAKE, REAL], Cols: Predicted [FAKE, REAL]):",
        str(np.array(best_eval["confusion_matrix"])),
        "",
        "Disclaimer:",
        DISCLAIMER_TEXT,
        "",
    ]
    with open(results_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def save_model(
    model: Any,
    vectorizer: TfidfVectorizer,
    model_path: Path = MODEL_PATH,
    vectorizer_path: Path = VECTORIZER_PATH,
) -> None:
    """Save the trained classifier and fitted TF-IDF vectorizer using joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved TF-IDF vectorizer to: {vectorizer_path}")


def load_model(
    model_path: Path = MODEL_PATH,
    vectorizer_path: Path = VECTORIZER_PATH,
) -> Tuple[Any, TfidfVectorizer]:
    """Load the persisted classifier and TF-IDF vectorizer from disk without retraining."""
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Saved model or vectorizer not found at '{model_path}' and '{vectorizer_path}'.\n"
            "Please run 'python fake_news_detector.py' first to train and save the model."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_news(
    article_text: str,
    model: Optional[Any] = None,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Dict[str, Any]:
    """
    Predict whether a news article is REAL or FAKE along with calibrated confidence.

    Validates user input, applies the exact same text cleaning pipeline, transforms
    with the fitted TF-IDF vectorizer, and computes class probabilities.
    """
    if article_text is None or not isinstance(article_text, str):
        raise ValueError("Input article must be a non-empty string.")

    cleaned = clean_text(article_text)
    if len(cleaned) < 10 or len(cleaned.split()) < 3:
        raise ValueError(
            "Please enter a valid news article or headline containing at least 3 meaningful words."
        )

    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    features = vectorizer.transform([cleaned])
    pred_id = int(model.predict(features)[0])
    pred_label = ID_TO_LABEL[pred_id]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[pred_id]) * 100.0
        prob_fake = float(probabilities[0]) * 100.0
        prob_real = float(probabilities[1]) * 100.0
    else:
        raise RuntimeError(
            "Loaded model does not support probability estimation. Ensure a calibrated classifier is used."
        )

    return {
        "prediction": pred_label,
        "confidence": round(confidence, 2),
        "probabilities": {
            "FAKE": round(prob_fake, 2),
            "REAL": round(prob_real, 2),
        },
        "disclaimer": DISCLAIMER_TEXT,
    }


def run_training_pipeline(dataset_dir: Path = DATASET_DIR) -> Dict[str, Any]:
    """
    Execute the end-to-end training and evaluation pipeline:
    1. Load dataset
    2. Clean and deduplicate data
    3. Stratified train/test split BEFORE TF-IDF fitting (prevents data leakage)
    4. Fit TF-IDF on training data only
    5. Train and compare classifiers on held-out test set
    6. Save best model, vectorizer, and evaluation artifacts
    """
    print("=" * 65)
    print("FAKE NEWS DETECTION SYSTEM - TRAINING PIPELINE")
    print("=" * 65)

    raw_df = load_dataset(dataset_dir=dataset_dir)
    df, prep_stats = prepare_data(raw_df)

    print(
        f"[INFO] Dataset prepared: {prep_stats['deduplicated_rows']} unique articles "
        f"(FAKE: {prep_stats['fake_count']}, REAL: {prep_stats['real_count']}, "
        f"Duplicates removed: {prep_stats['duplicates_removed']})"
    )

    X = df["combined_text"].values
    y = df["target"].values

    # Split data BEFORE fitting TF-IDF to strictly prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    split_stats = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    print(
        f"[INFO] Train/Test Split completed: {split_stats['train_size']} train, "
        f"{split_stats['test_size']} test"
    )

    # Fit TF-IDF ONLY on training data
    vectorizer = build_vectorizer()
    print("[INFO] Fitting TF-IDF vectorizer on training set only...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"[INFO] TF-IDF Vocabulary Size: {len(vectorizer.vocabulary_):,} features")

    best_model, best_name, evaluations = train_models(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )

    best_eval = evaluations[best_name]
    print("\n" + "-" * 65)
    print(f"FINAL EVALUATION ON HELD-OUT TEST SET ({best_name})")
    print("-" * 65)
    print(f"Accuracy : {best_eval['accuracy'] * 100:.2f}%")
    print(f"Precision: {best_eval['precision'] * 100:.2f}%")
    print(f"Recall   : {best_eval['recall'] * 100:.2f}%")
    print(f"F1-Score : {best_eval['f1_score'] * 100:.2f}%\n")
    print("Classification Report:")
    print(best_eval["classification_report"])
    print("Confusion Matrix (Rows: True [FAKE, REAL], Cols: Predicted [FAKE, REAL]):")
    print(np.array(best_eval["confusion_matrix"]))
    print("-" * 65)

    save_model(best_model, vectorizer)
    save_evaluation_artifacts(df, prep_stats, split_stats, best_name, evaluations)
    print(f"[INFO] Evaluation reports and charts saved to: {RESULTS_DIR}")

    return {
        "best_model": best_model,
        "vectorizer": vectorizer,
        "best_name": best_name,
        "evaluations": evaluations,
    }


def format_prediction_output(result: Dict[str, Any]) -> str:
    """Format prediction result for console output."""
    return (
        f"\nPrediction: {result['prediction']}\n"
        f"Confidence: {result['confidence']:.2f}%\n\n"
        f"{result['disclaimer']}\n"
    )


def interactive_console(model: Any, vectorizer: TfidfVectorizer) -> None:
    """Prompt the user to enter news articles for classification in the console."""
    print("\n" + "=" * 65)
    print("NEWS ARTICLE PREDICTION CONSOLE")
    print("Type or paste a news article and press Enter (or type 'exit' to quit).")
    print("=" * 65)

    while True:
        try:
            user_input = input("\nEnter news article:\n\n").strip()
        except EOFError:
            print("\n[INFO] End of input stream reached. Exiting console.")
            break
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user. Exiting console.")
            break

        if not user_input or user_input.lower() in {"exit", "quit", "q"}:
            print("[INFO] Exiting prediction console.")
            break

        try:
            res = predict_news(user_input, model=model, vectorizer=vectorizer)
            print(format_prediction_output(res))
        except ValueError as err:
            print(f"[WARNING] Invalid input: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fake News Detection System - Train models and classify news articles."
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Classify a provided news article string directly using the saved model.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training and load the existing saved model from models/.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Exit after training/prediction without launching interactive input loop.",
    )
    args = parser.parse_args()

    try:
        if args.predict is not None:
            model, vectorizer = load_model()
            result = predict_news(args.predict, model=model, vectorizer=vectorizer)
            print(format_prediction_output(result))
            return

        if args.no_train and MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            print("[INFO] Loading existing trained model and TF-IDF vectorizer...")
            model, vectorizer = load_model()
        else:
            pipeline_out = run_training_pipeline()
            model = pipeline_out["best_model"]
            vectorizer = pipeline_out["vectorizer"]

        if not args.no_interactive and sys.stdin.isatty():
            interactive_console(model, vectorizer)

    except FileNotFoundError as fnf_err:
        print(f"\n[ERROR] {fnf_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
