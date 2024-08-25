
from __future__ import annotations
import pandas as pd
import numpy as np

def compute_eui(df: pd.DataFrame, area_column: str = "area_m2") -> float:
    """Compute EUI = sum(kwh) / area."""
    if df.empty:
        return 0.0
    annual = float(df.get("kwh", pd.Series([0.0])).sum())
    area = float(df.get(area_column, pd.Series([np.nan])).iloc[0]) if area_column in df.columns else np.nan
    if np.isnan(area) or area <= 0:
        raise ValueError("Area is missing or non-positive")
    return annual / area

def baseline_delta(df: pd.DataFrame) -> float:
    """Compute (baseline - actual) over the dataset, 0 if no baseline."""
    if df.empty or "kwh" not in df.columns:
        return 0.0
    actual = float(df["kwh"].sum())
    if "baseline" not in df.columns:
        return 0.0
    base = float(df["baseline"].sum())
    return base - actual
