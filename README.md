PCOS Machine Learning Prediction

A reproducible three-class PCOS screening project using Random Forest and Support Vector Machine (SVM), served through an interactive Streamlit application.

Academic context: Developed as a Final Semester Examination (UAS) project for the Machine Learning / Pembelajaran Mesin course in Informatics Engineering at Universitas 17 Agustus 1945 Surabaya.

Medical disclaimer: This is an educational prototype, not a diagnostic device. Borderline and PCOS Positif are project-derived symptom-score labels and are not clinically validated severity categories. Predictions must not replace examination, laboratory interpretation, or advice from a qualified healthcare professional.

Highlights

Reproducible model training from the included Excel dataset

Three project classes: Tidak PCOS, Borderline, and PCOS Positif

Random Forest and RBF-kernel SVM comparison

Median imputation fitted only on training data

Standardization inside the SVM training pipeline

Stratified 80/20 train-test split

Probability estimates and downloadable CSV results

Polished Streamlit dashboard with class composition, confidence, held-out confusion matrix, and feature importance

Automated tests for preprocessing and feature alignment

Model performance

Metrics below were calculated on a held-out stratified test set of 109 rows using random state 42.

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

SVM

90.83%

92.44%

91.34%

80.48%

98.66%

SVM is the default model because it achieved the stronger weighted F1, macro F1, and ROC-AUC. Random Forest did not identify the small Borderline class in this particular test split, which is why accuracy alone should not be used to select the model.

Dataset

The included workbook contains:

541 rows

41 model features

364 Tidak PCOS records

31 derived Borderline records

146 derived PCOS Positif records

Target column: PCOS (Y/N)

Feature groups include demographic measurements, menstrual-cycle information, hormone measurements, physical symptoms, blood pressure, and follicle measurements. Identifier columns are excluded from training.

Derived three-class target

The original workbook provides a binary PCOS target. For the academic multiclass extension used in the project presentation, the target is reconstructed as follows:

Original PCOS target 0 → Tidak PCOS

Original PCOS target 1 with 0–2 selected symptoms → Borderline

Original PCOS target 1 with 3–6 selected symptoms → PCOS Positif

The six binary indicators are weight gain, hair growth, skin darkening, hair loss, pimples, and fast-food consumption. This is a project-defined grouping for machine-learning experimentation, not a validated clinical severity scale.

The repository copy does not currently document the dataset's original publication URL or license. Add the authoritative source and license before redistributing the dataset beyond this educational repository.

Top Random Forest features

The trained Random Forest ranks these among its most influential features:

Right follicle count

Left follicle count

Skin darkening

Weight gain

Hair growth

Fast-food frequency indicator

AMH

BMI

Waist-to-hip ratio

LH

Feature importance describes model behavior and must not be interpreted as medical causation.

Project structure

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
└── README.md

Installation

git clone https://github.com/bioonahnuu-design/PCOS_ML_Prediction.git
cd PCOS_ML_Prediction
python -m venv .venv

Windows PowerShell:

.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Run the Streamlit application

streamlit run app.py

Open http://localhost:8501, choose a model, and upload an Excel workbook containing a sheet named Full_new.

Reproduce model training

python train_models.py

This regenerates both model artifacts and metrics.json from the included dataset.

Run tests

python -m unittest discover -s tests -v

Limitations

The dataset is small and strongly class-imbalanced, especially the 31-row Borderline class.

The three-class target is derived from symptoms that are also model inputs, so reported performance measures reproducibility of the project rule—not independent clinical validity.

Results come from one stratified holdout split, not external clinical validation.

The application performs batch classification from a structured workbook.

Missing features are imputed and may reduce prediction reliability.

Model performance may not generalize to other populations or clinical settings.

Original dataset provenance and license still need authoritative documentation.

Roadmap

Add authoritative dataset source and license

Add cross-validation with confidence intervals

Add multiclass ROC-curve visualization

Add model-card documentation

Add a polished application screenshot

Publish the educational demo on Streamlit Community Cloud

Team — Kelompok 10

NBI

Name

1462400047

Hilva Najwa Aulia

1462400146

Nahnu Rohmania

1462400167

Zenicio Xavier Brito De Carvalho

Informatics Engineering — Universitas 17 Agustus 1945 Surabaya.

Project repository maintained by Nahnu Rohmania.
