"""Shared data preparation helpers for the PCOS prediction project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "PCOS (Y/N)"
DROP_COLUMNS = ("Sl. No", "Patient File No.", "Unnamed: 44")
SYMPTOM_COLUMNS = (
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
)


def read_dataset(source: str | Path | object) -> pd.DataFrame:
    """Read the named data sheet from the project Excel format."""
    return pd.read_excel(source, sheet_name="Full_new")


def clean_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and convert feature-like values to numeric."""
    df = raw_df.copy()
    df.columns = df.columns.astype(str).str.strip()
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = (
                df[column].astype(str).str.strip().str.replace(",", ".", regex=False)
                .replace({"nan": np.nan, "": np.nan, "None": np.nan})
            )
            converted = pd.to_numeric(df[column], errors="coerce")
            if converted.notna().sum() >= df[column].notna().sum() * 0.5:
                df[column] = converted
    return df


def create_multiclass_target(df: pd.DataFrame) -> pd.Series:
    """Derive the academic 0/1/2 target used by the project presentation."""
    if TARGET not in df.columns:
        raise ValueError(f"Required target column not found: {TARGET}")
    missing_symptoms = [name for name in SYMPTOM_COLUMNS if name not in df.columns]
    if missing_symptoms:
        raise ValueError("Missing symptom columns: " + ", ".join(missing_symptoms))
    binary_target = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not set(binary_target.unique()).issubset({0, 1}):
        raise ValueError("Original PCOS target must contain only labels 0 and 1")
    symptom_score = (
        df[list(SYMPTOM_COLUMNS)].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    )
    return pd.Series(
        np.where(binary_target == 0, 0, np.where(symptom_score <= 2, 1, 2)),
        index=df.index,
        name="PCOS Class",
    )


def split_features_target(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and the project-derived multiclass target."""
    df = clean_dataframe(raw_df)
    target = create_multiclass_target(df)
    df = df.drop(columns=[TARGET])
    features = df.drop(columns=list(DROP_COLUMNS), errors="ignore")
    return features.apply(pd.to_numeric, errors="coerce"), target


def prepare_prediction_features(
    raw_df: pd.DataFrame, expected_features: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Align uploaded data to the exact feature schema used during training."""
    df = clean_dataframe(raw_df)
    df = df.drop(columns=[TARGET, *DROP_COLUMNS], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    missing = [name for name in expected_features if name not in df.columns]
    extra = [name for name in df.columns if name not in expected_features]
    for name in missing:
        df[name] = np.nan
    return df[expected_features], missing, extra
