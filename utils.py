from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

ABC_DEFAULTS = {"A":70, "B":90}  # cumulative value thresholds (%)
XYZ_DEFAULTS = {"X":0.5, "Y":1.0}  # CV thresholds

def compute_abc_xyz(
    df: pd.DataFrame,
    month_cols: List[str],
    unit_cost_col: str = "UnitCost",
    abc_thresholds: Dict[str, float] = ABC_DEFAULTS,
    xyz_thresholds: Dict[str, float] = XYZ_DEFAULTS,
) -> pd.DataFrame:
    """Compute Annual Usage, Annual Value, ABC (by value), XYZ (by demand CV)."""
    df = df.copy()
    # Annual usage and value
    df["Annual_Usage"] = df[month_cols].sum(axis=1)
    df["Annual_Value"] = df["Annual_Usage"] * df[unit_cost_col]
    # ABC by cumulative value
    df = df.sort_values("Annual_Value", ascending=False).reset_index(drop=True)
    total_value = df["Annual_Value"].sum() if df["Annual_Value"].sum() != 0 else 1e-6
    df["CumPct"] = df["Annual_Value"].cumsum() / total_value * 100.0

    def _abc(c):
        if c <= abc_thresholds["A"]:
            return "A"
        elif c <= abc_thresholds["B"]:
            return "B"
        return "C"

    df["ABC"] = df["CumPct"].apply(_abc)

    # XYZ by coefficient of variation (std/mean)
    means = df[month_cols].mean(axis=1)
    stds = df[month_cols].std(axis=1, ddof=0)  # population std
    df["CV"] = stds / (means.replace(0, np.nan))
    df["CV"] = df["CV"].fillna(np.inf)

    def _xyz(cv):
        if cv < xyz_thresholds["X"]:
            return "X"
        elif cv < xyz_thresholds["Y"]:
            return "Y"
        return "Z"

    df["XYZ"] = df["CV"].apply(_xyz)
    df["Segment"] = df["ABC"] + "-" + df["XYZ"]
    return df

def heatmap_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot counts for ABC vs XYZ grid."""
    pivot = df.groupby(["ABC", "XYZ"]).size().unstack(fill_value=0)
    # Ensure all categories present
    for a in ["A","B","C"]:
        if a not in pivot.index: pivot.loc[a] = 0
    for x in ["X","Y","Z"]:
        if x not in pivot.columns: pivot[x] = 0
    pivot = pivot.loc[["A","B","C"], ["X","Y","Z"]]
    return pivot

def plot_heatmap(pivot: pd.DataFrame, title: str = "ABC × XYZ Segmentation"):
    """Simple matplotlib heatmap without specifying colors."""
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(pivot.values)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("XYZ (Demand Variability)")
    ax.set_ylabel("ABC (Value)")
    ax.set_title(title)
    # annotate counts
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, str(pivot.values[i, j]), ha="center", va="center")
    fig.tight_layout()
    return fig