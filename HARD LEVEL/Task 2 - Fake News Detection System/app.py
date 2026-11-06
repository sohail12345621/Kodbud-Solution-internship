"""
Streamlit Web Interface for the Fake News Detection System.
Run locally with:
    streamlit run app.py
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from fake_news_detector import (
    DISCLAIMER_TEXT,
    METRICS_JSON_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
    load_model,
    predict_news,
    run_training_pipeline,
)


@st.cache_resource
def get_cached_model():
    """Load and cache the trained classifier and TF-IDF vectorizer."""
    return load_model()


def load_evaluation_summary(metrics_path: Path = METRICS_JSON_PATH) -> Optional[Dict[str, Any]]:
    """Load saved evaluation metrics if available."""
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main() -> None:
    st.set_page_config(
        page_title="Fake News Detection System",
        layout="centered",
    )

    st.title("Fake News Detection System")
    st.markdown(
        """
        This application uses classical Natural Language Processing (**TF-IDF Vectorization**)
        and supervised **Machine Learning** to classify news articles as **REAL** or **FAKE**
        based on linguistic and lexical patterns learned from labeled news datasets.
        """
    )

    # Sidebar / Model Information Section
    st.sidebar.header("Model Information")
    metrics_data = load_evaluation_summary()

    if metrics_data:
        st.sidebar.markdown(f"**Selected Classifier:** {metrics_data.get('best_model', 'N/A')}")
        best_m = metrics_data.get("best_metrics", {})
        if best_m:
            st.sidebar.metric("Test Accuracy", f"{best_m.get('accuracy', 0) * 100:.2f}%")
            st.sidebar.metric("Test Precision", f"{best_m.get('precision', 0) * 100:.2f}%")
            st.sidebar.metric("Test Recall", f"{best_m.get('recall', 0) * 100:.2f}%")
            st.sidebar.metric("Test F1-Score", f"{best_m.get('f1_score', 0) * 100:.2f}%")

        prep = metrics_data.get("preprocessing_stats", {})
        if prep:
            st.sidebar.caption(
                f"Trained on {prep.get('deduplicated_rows', 0):,} unique articles "
                f"(FAKE: {prep.get('fake_count', 0):,}, REAL: {prep.get('real_count', 0):,})."
            )
    else:
        st.sidebar.info(
            "Model evaluation metrics not found. Run `python fake_news_detector.py` to train and evaluate models."
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Limitation Warning")
    st.sidebar.warning(DISCLAIMER_TEXT)

    # Verify model files exist
    if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
        st.error(
            "Trained model files were not found in `models/`. "
            "Please run `python fake_news_detector.py` first or click the button below if the dataset is placed in `dataset/`."
        )
        if st.button("Train Model Now"):
            with st.spinner("Training and evaluating models on dataset..."):
                try:
                    run_training_pipeline()
                    st.success("Model training completed. Reloading application...")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        return

    # Input Section
    st.subheader("Analyze a News Article")
    headline_input = st.text_input(
        "Article Headline (Optional):",
        placeholder="Enter the headline of the news article...",
    )
    article_input = st.text_area(
        "Enter news article:",
        height=240,
        placeholder="Paste the full text of the news article here...",
    )

    if st.button("Predict", type="primary"):
        combined_input = f"{headline_input.strip()} {article_input.strip()}".strip()
        if not combined_input:
            st.warning("Please enter a news article or headline before clicking Predict.")
        else:
            try:
                model, vectorizer = get_cached_model()
                result = predict_news(combined_input, model=model, vectorizer=vectorizer)

                prediction = result["prediction"]
                confidence = result["confidence"]
                probs = result["probabilities"]

                st.markdown("---")
                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)
                with col1:
                    if prediction == "REAL":
                        st.success(f"Prediction: {prediction}")
                    else:
                        st.error(f"Prediction: {prediction}")
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")

                st.write(
                    f"**Class Probabilities:** REAL: `{probs['REAL']:.2f}%` | FAKE: `{probs['FAKE']:.2f}%`"
                )

                st.info(result["disclaimer"])

            except ValueError as val_err:
                st.warning(str(val_err))
            except Exception as err:
                st.error(f"Error during prediction: {err}")

    st.markdown("---")
    st.caption(DISCLAIMER_TEXT)


if __name__ == "__main__":
    main()
