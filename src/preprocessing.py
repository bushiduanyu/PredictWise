"""Reusable feature engineering for PredictWise."""

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features used by the training notebook."""
    df = data.copy()

    missing = [column for column in NUMERIC_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df["Temperature difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    df["Power proxy [W]"] = (
        df["Torque [Nm]"]
        * df["Rotational speed [rpm]"]
        * 2
        * np.pi
        / 60
    )
    df["Tool wear level"] = pd.cut(
        df["Tool wear [min]"],
        bins=[-1, 80, 160, 260],
        labels=["Low", "Medium", "High"],
    )

    return df


def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categories with the same reference levels as training."""
    df = data.copy()

    if "Type" not in df.columns:
        raise ValueError("Missing required column: Type")

    # Explicit categories keep encoding stable even for a single app input row.
    df["Type"] = pd.Categorical(df["Type"], categories=["H", "L", "M"])
    df["Tool wear level"] = pd.Categorical(
        df["Tool wear level"],
        categories=["Low", "Medium", "High"],
        ordered=True,
    )

    return pd.get_dummies(
        df,
        columns=["Type", "Tool wear level"],
        drop_first=True,
        dtype=int,
    )


def prepare_model_input(
    data: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Engineer, encode, and align input data to the trained model schema."""
    engineered = add_engineered_features(data)
    encoded = encode_categorical_features(engineered)
    return encoded.reindex(columns=list(feature_columns), fill_value=0)
