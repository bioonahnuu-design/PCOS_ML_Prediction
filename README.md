🧬 PCOS Multiclass Prediction

Machine Learning Dashboard with Random Forest & Support Vector Machine

An educational three-class PCOS prediction project built with Python, scikit-learn, and Streamlit.

541 records · 41 features · 3 project classes · 2 ML models

</div>

✨ Project at a Glance

Problem

Extend an academic binary PCOS prediction task into a three-class ML experiment

Classes

🟢 Tidak PCOS · 🟠 Borderline · 🔴 PCOS Positif

Models

Random Forest and RBF-kernel Support Vector Machine

Dataset

541 rows, 41 model features

Best model

SVM — 90.83% accuracy, 91.34% weighted F1, 80.48% macro F1

Interface

Interactive Streamlit dashboard with batch Excel upload

Context

Final Semester Examination (UAS) — Pembelajaran Mesin / Machine Learning

[!IMPORTANT]This repository is an educational machine-learning prototype, not a diagnostic device. Borderline and PCOS Positif are project-derived labels and are not clinically validated severity categories.

🎯 What the App Does

The application accepts a compatible Excel workbook and runs batch predictions using the selected classifier. It provides:

Three-class prediction: Tidak PCOS, Borderline, and PCOS Positif

Random Forest / SVM model selection

Prediction confidence for every processed row

Class-composition visualization

Held-out test metrics

Multiclass confusion matrix

Random Forest feature-importance visualization

Data explorer and downloadable CSV results

🧠 Machine Learning Workflow

flowchart LR
A[Excel Dataset] --> B[Cleaning & Feature Alignment]
B --> C[Derived 3-Class Target]
C --> D[Stratified 80/20 Split]
D --> E[Random Forest]
D --> F[RBF SVM]
E --> G[Evaluation]
F --> G
G --> H[Streamlit Dashboard]
H --> I[CSV Export]

📊 Model Performance

Metrics were calculated on a held-out stratified test set of 109 rows with random_state=42.

Model

Accuracy

Weighted Precision

Weighted F1

Macro F1

Weighted ROC-AUC

Random Forest

90.83%

85.82%

88.25%

61.95%

98.28%

Support Vector Machine

90.83%

92.44%

91.34%

80.48%

98.66%

Why SVM is the default

Both models reached the same overall accuracy, but SVM produced stronger Weighted F1, Macro F1, and ROC-AUC. Macro F1 is especially relevant here because the derived Borderline class is much smaller than the other classes.

Random Forest remains available because its feature-importance output makes model behavior easier to inspect.

🗂️ Dataset

The included workbook contains:

541 records

41 model features

364 original non-PCOS records

31 derived Borderline records

146 derived PCOS Positif records

Original target column: PCOS (Y/N)

Feature groups include demographic measurements, menstrual-cycle information, hormone measurements, physical symptoms, blood pressure, and follicle measurements. Identifier columns are excluded from training.

Three-class academic extension

The source workbook contains a binary PCOS target. For the multiclass experiment used in the project presentation, the target is reconstructed as:

Project class

Rule used in this project

🟢 Tidak PCOS

Original PCOS target = 0

🟠 Borderline

Original target = 1 and selected symptom score = 0–2

🔴 PCOS Positif

Original target = 1 and selected symptom score = 3–6

The six binary indicators are weight gain, hair growth, skin darkening, hair loss, pimples, and fast-food consumption.

This rule is an academic grouping for machine-learning experimentation. It must not be interpreted as a validated clinical PCOS severity scale.

🔎 Random Forest Model Insights

The current Random Forest ranks these among its most influential features:

Right follicle count

Left follicle count

Hair growth

Skin darkening

Fast-food indicator

AMH

Average right follicle size

Waist-to-hip ratio

Weight gain

Average left follicle size

Feature importance describes model behavior, not medical causation.

🧰 Tech Stack

Layer

Technology

Language

Python 3.11+

ML

scikit-learn 1.8

Data

pandas, NumPy, openpyxl

Visualization

Matplotlib, Seaborn

Interface

Streamlit

Model artifacts

joblib

Testing

unittest

📁 Project Structure

PCOS_ML_Prediction/
├── .devcontainer/
├── tests/
│ └── test_pcos_utils.py
├── PCOS_data_without_infertility.xlsx
├── app.py
├── metrics.json
├── pcos_utils.py
├── random_forest_pcos.pkl
├── svm_pcos.pkl
├── train_models.py
├── requirements.txt
├── LICENSE
└── README.md

🚀 Run Locally

git clone https://github.com/bioonahnuu-design/PCOS_ML_Prediction.git
cd PCOS_ML_Prediction
python -m venv .venv

Windows PowerShell:

.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py

Open http://localhost:8501, choose a classifier, and upload a compatible .xlsx workbook containing a Full_new sheet.

♻️ Reproduce Training

python train_models.py

This regenerates the Random Forest and SVM model artifacts together with metrics.json.

✅ Run Tests

python -m unittest discover -s tests -v

The tests cover multiclass target construction, required-target validation, and prediction feature alignment.

⚠️ Limitations

The dataset is small and class-imbalanced, especially the derived Borderline class.

The three-class target is derived from symptoms that are also used as model inputs.

Metrics therefore measure reproducibility of the academic project rule, not independent clinical validity.

Evaluation currently uses one stratified hold-out split rather than external clinical validation.

Missing input features are imputed and may reduce reliability.

The original dataset publication URL and license are not yet documented in this repository.

🛣️ Roadmap

Add authoritative dataset source and license

Add cross-validation with confidence intervals

Add multiclass ROC-curve visualization

Add model-card documentation

Add a dashboard screenshot to this README

🎓 Academic Context

This project was developed as a Final Semester Examination (UAS) project for the Pembelajaran Mesin (Machine Learning) course, Informatics Engineering, Universitas 17 Agustus 1945 Surabaya.

👥 Team — Kelompok 10

NBI

Name

1462400047

Hilva Najwa Aulia

1462400146

Nahnu Rohmania

1462400167

Zenicio Xavier Brito De Carvalho

Repository maintained by @bioonahnuu-design.

<div align="center">

Built for learning, reproducibility, and responsible machine-learning demonstration.

</div>
