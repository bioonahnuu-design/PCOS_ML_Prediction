"""Streamlit dashboard for the reproducible three-class PCOS classifiers."""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from pcos_utils import prepare_prediction_features, read_dataset

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Random Forest": BASE_DIR / "random_forest_pcos.pkl",
    "Support Vector Machine": BASE_DIR / "svm_pcos.pkl",
}
LABELS = {0: "Tidak PCOS", 1: "Borderline", 2: "PCOS Positif"}
CLASS_ORDER = list(LABELS.values())
CLASS_COLORS = ["#22c55e", "#f59e0b", "#ef4444"]

st.set_page_config(
    page_title="PCOS ML Screening Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f8fafc; }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    div[data-testid="stSidebarContent"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid #dbe7e5 !important;
        box-shadow: 8px 0 24px rgba(15, 23, 42, .035) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    section[data-testid="stSidebar"] small {
        color: #64748b !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #e2e8f0 !important;
    }
    .block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1450px; }
    .hero {
        padding: 1.65rem 1.8rem; border-radius: 22px; color: #0f172a;
        background: linear-gradient(120deg, #ffffff 0%, #f0fdfa 55%, #ccfbf1 100%);
        border: 1px solid #cce8e3;
        box-shadow: 0 12px 30px rgba(15, 118, 110, .08); margin-bottom: 1.1rem;
    }
    .hero h1 { margin: 0; color:#0f172a; font-size: 2.15rem; letter-spacing: -.035em; }
    .hero p { margin: .45rem 0 0; color: #475569; font-size: 1rem; }
    .eyebrow { color: #0f766e; font-size: .76rem; font-weight: 800; letter-spacing: .14em; }
    .class-card {
        padding: 1.05rem 1.2rem; border: 1px solid #e2e8f0; border-radius: 16px;
        background: white; box-shadow: 0 5px 18px rgba(15,23,42,.05); min-height: 116px;
    }
    .class-card .label { color: #64748b; font-size: .84rem; font-weight: 700; }
    .class-card .value { color: #0f172a; font-size: 2rem; font-weight: 800; line-height: 1.2; }
    .class-card .share { color: #64748b; font-size: .78rem; }
    .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 7px; }
    .section-title { color:#0f172a; font-weight:800; font-size:1.2rem; margin:.2rem 0 .15rem; }
    .section-copy { color:#64748b; margin-bottom:.75rem; font-size:.9rem; }
    [data-testid="stMetric"] {
        background:white; border:1px solid #e2e8f0; padding:.85rem 1rem;
        border-radius:14px; box-shadow:0 4px 14px rgba(15,23,42,.04);
    }
    [data-testid="stMetricLabel"] * { color:#64748b !important; }
    [data-testid="stMetricValue"] * { color:#0f172a !important; }
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #f8fafc !important;
        border: 1px solid #dbe7e5 !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent !important;
        border: 0 !important;
        padding: .15rem 0 !important;
    }
    div[role="radiogroup"] { gap: .55rem; }
    div[role="radiogroup"] label {
        background: #ffffff; border: 1px solid #dbe7e5; border-radius: 12px;
        padding: .55rem .8rem; transition: all .15s ease;
    }
    div[data-testid="stDataFrame"] { border:1px solid #e2e8f0; border-radius:14px; overflow:hidden; }
    .medical-note {
        padding:.8rem 1rem; border-radius:12px; border-left:4px solid #f59e0b;
        background:#fffbeb; color:#78350f; font-size:.86rem; margin-bottom:1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    metrics = json.loads((BASE_DIR / "metrics.json").read_text(encoding="utf-8"))
    return models, metrics


def render_class_card(label: str, value: int, total: int, color: str) -> None:
    share = (value / total * 100) if total else 0
    st.markdown(
        f"""<div class="class-card">
        <div class="label"><span class="dot" style="background:{color}"></span>{label}</div>
        <div class="value">{value:,}</div>
        <div class="share">{share:.1f}% dari {total:,} baris</div>
        </div>""",
        unsafe_allow_html=True,
    )


def style_figure(figure: plt.Figure) -> None:
    figure.patch.set_facecolor("#ffffff")
    figure.tight_layout()


st.markdown(
    """<div class="hero">
    <div class="eyebrow">MACHINE LEARNING • EDUCATIONAL SCREENING</div>
    <h1>PCOS Multiclass Prediction Dashboard</h1>
    <p>Explore three-class predictions with Random Forest &amp; Support Vector Machine.</p>
    </div>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div class="medical-note"><b>Educational prototype.</b> “Borderline” and
    “PCOS Positif” are project-derived labels for ML experimentation, not clinically
    validated diagnoses or severity stages. Results must not replace professional
    healthcare evaluation.</div>""",
    unsafe_allow_html=True,
)

try:
    models, metrics = load_artifacts()
except Exception as exc:
    st.error(f"Model artifacts could not be loaded: {exc}")
    st.stop()

st.sidebar.markdown("## 🧬 PCOS ML")
st.sidebar.caption("Multiclass prediction workspace")
st.sidebar.markdown("---")
st.sidebar.markdown("### Model configuration")
model_name = st.sidebar.radio("Choose classifier", list(MODEL_PATHS), index=1)
selected_model = models[model_name]
metric_key = "random_forest" if model_name == "Random Forest" else "svm"
model_metrics = metrics["models"][metric_key]
st.sidebar.markdown("### Held-out performance")
st.sidebar.metric("Accuracy", f"{model_metrics['accuracy']:.2%}")
st.sidebar.metric("Weighted F1", f"{model_metrics['f1_weighted']:.2%}")
st.sidebar.metric("Macro F1", f"{model_metrics['f1_macro']:.2%}")
st.sidebar.caption(
    f"Stratified test set · {metrics['test_rows']} rows · random state {metrics['random_state']}"
)
st.sidebar.markdown("---")
st.sidebar.caption("Nahnu Rohmania · Informatics Engineering")

st.markdown('<div class="section-title">01 · Upload dataset</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Use the project workbook or another compatible .xlsx file with a Full_new sheet.</div>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Upload PCOS workbook", type=["xlsx"], label_visibility="collapsed"
)
if uploaded_file is None:
    st.info(
        "Upload an Excel workbook to start. The repository includes "
        "PCOS_data_without_infertility.xlsx as the project sample."
    )
    st.stop()

try:
    raw_df = read_dataset(uploaded_file)
    expected_features = list(selected_model.feature_names_in_)
    features, missing, extra = prepare_prediction_features(raw_df, expected_features)
    predictions = selected_model.predict(features).astype(int)
    probabilities = selected_model.predict_proba(features)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

if missing:
    st.warning("Missing features were imputed: " + ", ".join(missing))
if extra:
    st.caption(f"{len(extra)} extra column(s) were ignored during model inference.")

result_df = raw_df.copy()
result_df["Prediction"] = pd.Series(predictions, index=result_df.index).map(LABELS)
result_df["Confidence"] = probabilities.max(axis=1)
counts = result_df["Prediction"].value_counts()
total_rows = len(result_df)

st.markdown('<div class="section-title">02 · Screening overview</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-copy">Batch results from <b>{model_name}</b> · {total_rows:,} rows processed.</div>',
    unsafe_allow_html=True,
)
cards = st.columns(3)
for column, label, color in zip(cards, CLASS_ORDER, CLASS_COLORS):
    with column:
        render_class_card(label, int(counts.get(label, 0)), total_rows, color)

overview_tab, insight_tab, data_tab = st.tabs(
    ["◉ Prediction overview", "⌁ Model insights", "▦ Data explorer"]
)

with overview_tab:
    st.write("")
    chart_col, table_col = st.columns([0.9, 1.55], gap="large")
    with chart_col:
        st.markdown("#### Class composition")
        st.caption("Share of model predictions across all uploaded rows")
        values = [int(counts.get(label, 0)) for label in CLASS_ORDER]
        figure, axis = plt.subplots(figsize=(5.2, 4.2))
        wedges, _ = axis.pie(
            values,
            startangle=90,
            colors=CLASS_COLORS,
            wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 3},
        )
        axis.text(0, 0.08, f"{total_rows:,}", ha="center", va="center", fontsize=24, fontweight="bold")
        axis.text(0, -0.12, "rows", ha="center", va="center", fontsize=10, color="#64748b")
        axis.legend(wedges, CLASS_ORDER, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)
        style_figure(figure)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    with table_col:
        st.markdown("#### Prediction sample")
        st.caption("First 20 rows with predicted class and model confidence")
        preview = result_df[["Prediction", "Confidence"]].head(20).copy()
        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=False,
            column_config={
                "Prediction": st.column_config.TextColumn("Predicted class"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.1%%"
                ),
            },
        )
        st.caption(
            "Confidence is the model's highest predicted class probability; it is not clinical certainty."
        )

with insight_tab:
    st.write("")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{model_metrics['accuracy']:.2%}")
    metric_cols[1].metric("Weighted F1", f"{model_metrics['f1_weighted']:.2%}")
    metric_cols[2].metric("Macro F1", f"{model_metrics['f1_macro']:.2%}")
    metric_cols[3].metric("ROC-AUC OvR", f"{model_metrics['roc_auc_ovr_weighted']:.2%}")

    matrix_col, importance_col = st.columns(2, gap="large")
    with matrix_col:
        st.markdown("#### Held-out confusion matrix")
        st.caption("Rows = actual project class · columns = predicted project class")
        matrix = np.asarray(model_metrics["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(6.5, 4.8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=sns.light_palette("#0f766e", as_cmap=True),
            cbar=False,
            xticklabels=CLASS_ORDER,
            yticklabels=CLASS_ORDER,
            ax=axis,
        )
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.tick_params(axis="x", rotation=20)
        axis.tick_params(axis="y", rotation=0)
        style_figure(figure)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    with importance_col:
        st.markdown("#### Model interpretation")
        if model_name == "Random Forest":
            classifier = selected_model.named_steps["model"]
            importance = (
                pd.DataFrame({"Feature": expected_features, "Importance": classifier.feature_importances_})
                .sort_values("Importance", ascending=False)
                .head(10)
                .sort_values("Importance")
            )
            st.caption("Top 10 Random Forest feature importances")
            figure, axis = plt.subplots(figsize=(6.5, 4.8))
            axis.barh(importance["Feature"], importance["Importance"], color="#0f766e")
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.grid(axis="x", color="#e2e8f0", linewidth=.8)
            axis.set_axisbelow(True)
            axis.set_xlabel("Importance")
            axis.set_ylabel("")
            style_figure(figure)
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)
        else:
            st.caption("Why no feature-importance chart is shown for the RBF SVM")
            st.info(
                "The selected SVM uses a non-linear RBF kernel, so it does not expose "
                "direct per-feature coefficients. Switch to Random Forest to inspect "
                "its feature-importance ranking."
            )
            st.markdown(
                "**Why SVM is the default:** it achieved stronger Macro F1 on the held-out "
                "split, which matters because the Borderline class is small."
            )

with data_tab:
    st.write("")
    st.markdown("#### Uploaded dataset + predictions")
    st.caption("Inspect the processed batch before exporting the result.")
    st.dataframe(result_df.head(100), use_container_width=True, hide_index=True)
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "⬇ Download prediction results (.csv)",
        data=csv_buffer.getvalue(),
        file_name="pcos_multiclass_prediction_results.csv",
        mime="text/csv",
        use_container_width=False,
    )

st.caption(
    "PCOS ML Prediction · Educational machine-learning portfolio project · "
    "Three project-defined classes: Tidak PCOS, Borderline, PCOS Positif"
)