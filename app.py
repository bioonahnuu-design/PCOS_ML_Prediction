import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="PCOS Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_PATH = os.path.join(BASE_DIR, "random_forest_pcos.pkl")
SVM_PATH = os.path.join(BASE_DIR, "svm_pcos.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

LABEL_MAP = {
    0: "Tidak PCOS",
    1: "PCOS Ringan",
    2: "PCOS Berat"
}

COLUMNS_TO_DROP = ["Sl. No", "Patient File No.", "Unnamed: 44"]


# =========================================================
# FUNGSI BANTUAN
# =========================================================
@st.cache_resource
def load_artifacts():
    errors = []
    rf_model = None
    svm_model = None
    scaler = None

    if os.path.exists(RF_PATH):
        try:
            rf_model = joblib.load(RF_PATH)
        except Exception as e:
            errors.append(f"Gagal memuat random_forest_pcos.pkl: {e}")
    else:
        errors.append("File random_forest_pcos.pkl tidak ditemukan di folder aplikasi.")

    if os.path.exists(SVM_PATH):
        try:
            svm_model = joblib.load(SVM_PATH)
        except Exception as e:
            errors.append(f"Gagal memuat svm_pcos.pkl: {e}")
    else:
        errors.append("File svm_pcos.pkl tidak ditemukan di folder aplikasi.")

    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            errors.append(f"Gagal memuat scaler.pkl: {e}")
    else:
        errors.append("File scaler.pkl tidak ditemukan di folder aplikasi.")

    return rf_model, svm_model, scaler, errors


def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Strip nama kolom
    df.columns = df.columns.astype(str).str.strip()

    # Hapus kolom yang tidak diperlukan
    for col in COLUMNS_TO_DROP:
        col_stripped = col.strip()
        matching = [c for c in df.columns if c.strip() == col_stripped]
        for m in matching:
            df.drop(columns=[m], inplace=True, errors="ignore")

    # Ubah koma menjadi titik pada kolom bertipe object
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = df[col].replace({"nan": np.nan, "": np.nan, "None": np.nan})

    # Convert numeric untuk seluruh kolom yang memungkinkan
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Jika sebagian besar nilai berhasil dikonversi, gunakan versi numerik
        non_null_original = df[col].notna().sum()
        non_null_converted = converted.notna().sum()
        if non_null_original == 0 or non_null_converted >= non_null_original * 0.5:
            df[col] = converted

    # Hitung ulang BMI jika kolom Weight dan Height tersedia
    weight_col = None
    height_col = None
    bmi_col = None

    for c in df.columns:
        c_lower = c.lower()
        if "weight" in c_lower and "kg" in c_lower:
            weight_col = c
        if "height" in c_lower:
            height_col = c
        if c_lower.strip() == "bmi":
            bmi_col = c

    if weight_col is not None and height_col is not None:
        height_m = df[height_col] / 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            recalculated_bmi = df[weight_col] / (height_m ** 2)
        if bmi_col is not None:
            df[bmi_col] = recalculated_bmi
        else:
            df["BMI"] = recalculated_bmi

    # Isi missing value numerik dengan median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Isi missing value kategorikal dengan modus
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            if len(mode_series) > 0:
                df[col] = df[col].fillna(mode_series.iloc[0])

    return df


def align_features(df: pd.DataFrame, reference_features):
    """
    Menyusun ulang dan menyaring kolom dataframe agar sama persis
    dengan urutan dan nama fitur yang diharapkan oleh model/scaler.
    """
    missing_features = [f for f in reference_features if f not in df.columns]
    extra_features = [f for f in df.columns if f not in reference_features]

    # Tambahkan kolom yang hilang dengan nilai 0 agar tidak error
    for f in missing_features:
        df[f] = 0

    aligned_df = df[list(reference_features)]
    return aligned_df, missing_features, extra_features


def get_reference_features(scaler, model):
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None


def find_target_like_columns(df: pd.DataFrame):
    target_cols = []
    for c in df.columns:
        c_lower = c.lower()
        if "pcos" in c_lower:
            target_cols.append(c)
    return target_cols


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h1>🩺 PCOS Prediction System</h1>
        <h4 style="color:gray; font-weight:400;">
            Prediksi Tingkat Keparahan PCOS Menggunakan Machine Learning
        </h4>
    </div>
    """,
    unsafe_allow_html=True
)

rf_model, svm_model, scaler, load_errors = load_artifacts()

if load_errors:
    for err in load_errors:
        st.error(err)
    st.warning(
        "Pastikan file random_forest_pcos.pkl, svm_pcos.pkl, dan scaler.pkl "
        "berada pada folder yang sama dengan app.py sebelum menjalankan prediksi."
    )

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Pengaturan")
model_choice = st.sidebar.radio(
    "Pilih Model",
    options=["Random Forest", "Support Vector Machine"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Informasi")
st.sidebar.info(
    "Upload dataset PCOS dalam format .xlsx pada panel utama, "
    "lalu sistem akan melakukan preprocessing dan prediksi secara otomatis."
)

selected_model = rf_model if model_choice == "Random Forest" else svm_model
selected_model_name = "Random Forest" if model_choice == "Random Forest" else "Support Vector Machine (SVM)"

# =========================================================
# UPLOAD DATASET
# =========================================================
st.markdown("## 📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload file dataset (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_excel(uploaded_file, sheet_name=1)
    except Exception as e:
        st.error(f"Gagal membaca sheet ke-2 dari file yang diupload: {e}")
        st.stop()

    with st.spinner("Melakukan preprocessing data..."):
        try:
            df_clean = clean_dataset(raw_df)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat preprocessing data: {e}")
            st.stop()

    st.success("Dataset berhasil diupload dan diproses!")

    # =========================================================
    # PREVIEW DATASET
    # =========================================================
    st.markdown("## 👀 Preview Dataset")
    with st.container():
        st.dataframe(df_clean.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jumlah Data", f"{df_clean.shape[0]}")
        with col2:
            st.metric("Jumlah Feature", f"{df_clean.shape[1]}")

    # =========================================================
    # VALIDASI MODEL & SCALER
    # =========================================================
    if selected_model is None or scaler is None:
        st.error(
            "Model atau scaler belum berhasil dimuat. "
            "Prediksi tidak dapat dilanjutkan."
        )
        st.stop()

    # =========================================================
    # SIAPKAN FITUR SESUAI MODEL
    # =========================================================
    df_features = df_clean.copy()

    target_like_cols = find_target_like_columns(df_features)
    for c in target_like_cols:
        df_features.drop(columns=[c], inplace=True, errors="ignore")

    # Hanya gunakan kolom numerik untuk proses scaling & prediksi
    df_features = df_features.select_dtypes(include=[np.number])

    reference_features = get_reference_features(scaler, selected_model)

    if reference_features is None:
        st.warning(
            "Model/scaler tidak menyimpan informasi nama fitur (feature_names_in_). "
            "Sistem akan menggunakan seluruh kolom numerik yang tersedia sesuai urutan dataset."
        )
        X = df_features
    else:
        X, missing_feats, extra_feats = align_features(df_features, reference_features)
        if missing_feats:
            st.warning(
                "Beberapa fitur yang dibutuhkan model tidak ditemukan pada dataset "
                f"dan diisi dengan nilai 0: {', '.join(missing_feats)}"
            )

    # =========================================================
    # SCALING & PREDIKSI
    # =========================================================
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan scaling data: {e}")
        st.stop()

    try:
        predictions = selected_model.predict(X_scaled)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.stop()

    predictions = np.array(predictions).astype(int)
    prediction_labels = pd.Series(predictions).map(LABEL_MAP)

    result_df = df_clean.copy()
    result_df["Prediction"] = prediction_labels.values

    st.markdown("---")
    st.markdown(f"## 🔮 Hasil Prediksi — Model: **{selected_model_name}**")

    # =========================================================
    # RINGKASAN JUMLAH KELAS
    # =========================================================
    counts = prediction_labels.value_counts()
    tidak_pcos = int(counts.get("Tidak PCOS", 0))
    pcos_ringan = int(counts.get("PCOS Ringan", 0))
    pcos_berat = int(counts.get("PCOS Berat", 0))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Tidak PCOS", tidak_pcos)
    with col2:
        st.metric("🟡 PCOS Ringan", pcos_ringan)
    with col3:
        st.metric("🔴 PCOS Berat", pcos_berat)

    if pcos_berat > 0:
        st.warning(f"Terdapat {pcos_berat} data dengan prediksi PCOS Berat. Perlu perhatian lebih lanjut.")
    else:
        st.success("Tidak ditemukan data dengan prediksi PCOS Berat.")

    # =========================================================
    # VISUALISASI
    # =========================================================
    st.markdown("### 📊 Visualisasi Distribusi Hasil Prediksi")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("**Pie Chart — Distribusi Hasil Prediksi**")
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        labels_order = ["Tidak PCOS", "PCOS Ringan", "PCOS Berat"]
        values_order = [tidak_pcos, pcos_ringan, pcos_berat]
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

        non_zero_labels = [l for l, v in zip(labels_order, values_order) if v > 0]
        non_zero_values = [v for v in values_order if v > 0]
        non_zero_colors = [c for c, v in zip(colors, values_order) if v > 0]

        if len(non_zero_values) > 0:
            ax_pie.pie(
                non_zero_values,
                labels=non_zero_labels,
                autopct="%1.1f%%",
                colors=non_zero_colors,
                startangle=90
            )
            ax_pie.axis("equal")
            st.pyplot(fig_pie)
        else:
            st.info("Tidak ada data untuk ditampilkan pada pie chart.")

    with viz_col2:
        st.markdown("**Bar Chart — Distribusi Kelas**")
        fig_bar, ax_bar = plt.subplots(figsize=(5, 5))
        sns.barplot(
            x=labels_order,
            y=values_order,
            palette=colors,
            ax=ax_bar
        )
        ax_bar.set_xlabel("Kelas")
        ax_bar.set_ylabel("Jumlah")
        ax_bar.set_title("Distribusi Kelas Prediksi")
        for i, v in enumerate(values_order):
            ax_bar.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
        st.pyplot(fig_bar)

    # =========================================================
    # FEATURE IMPORTANCE (RANDOM FOREST)
    # =========================================================
    if model_choice == "Random Forest" and hasattr(selected_model, "feature_importances_"):
        st.markdown("### 🌲 Feature Importance — Top 10 (Random Forest)")

        try:
            if reference_features is not None:
                feat_names = reference_features
            else:
                feat_names = list(X.columns)

            importances = selected_model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": feat_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=fi_df,
                x="Importance",
                y="Feature",
                palette="viridis",
                ax=ax_fi
            )
            ax_fi.set_title("Top 10 Feature Importance")
            st.pyplot(fig_fi)
        except Exception as e:
            st.warning(f"Tidak dapat menampilkan feature importance: {e}")

    # =========================================================
    # TABEL HASIL & DOWNLOAD
    # =========================================================
    st.markdown("### 📋 Tabel Hasil Prediksi")
    with st.container():
        st.dataframe(result_df, use_container_width=True)

        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="⬇️ Download Hasil Prediksi (CSV)",
            data=csv_data,
            file_name="hasil_prediksi_pcos.csv",
            mime="text/csv"
        )

else:
    st.info("Silakan upload dataset dalam format .xlsx untuk memulai prediksi.")

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:13px;">
        PCOS Prediction System — Tugas Akhir Mata Kuliah Pemelajaran Mesin
    </div>
    """,
    unsafe_allow_html=True
)
