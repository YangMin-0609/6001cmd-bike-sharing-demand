"""Reproducible first-stage audit for the UCI Bike Sharing hour dataset.

Run from any directory with:
    .venv\\Scripts\\python.exe src\\analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hour.csv"
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"

REQUIRED_COLUMNS = {
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the hourly data and construct a true hourly timestamp."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download hour.csv from the UCI source."
        )

    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    data["timestamp"] = pd.to_datetime(data["dteday"]) + pd.to_timedelta(
        data["hr"], unit="h"
    )
    return data


def audit_data(data: pd.DataFrame) -> dict[str, object]:
    """Calculate the checks that should be discussed in Tasks 1, 3, and 4."""
    expected_hours = pd.date_range(
        data["timestamp"].min(), data["timestamp"].max(), freq="h"
    )
    missing_timestamps = expected_hours.difference(data["timestamp"])

    return {
        "rows": len(data),
        "original_columns": len(data.columns) - 1,
        "start_timestamp": data["timestamp"].min(),
        "end_timestamp": data["timestamp"].max(),
        "total_missing_values": int(data.isna().sum().sum()),
        "exact_duplicate_rows": int(data.drop(columns="timestamp").duplicated().sum()),
        "duplicate_timestamps": int(data["timestamp"].duplicated().sum()),
        "missing_hourly_timestamps": len(missing_timestamps),
        "target_identity_holds": bool(
            (data["cnt"] == data["casual"] + data["registered"]).all()
        ),
        "cnt_mean": data["cnt"].mean(),
        "cnt_median": data["cnt"].median(),
        "cnt_min": data["cnt"].min(),
        "cnt_max": data["cnt"].max(),
        "cnt_skewness": data["cnt"].skew(),
        "temp_atemp_correlation": data[["temp", "atemp"]].corr().iloc[0, 1],
    }


def export_tables(data: pd.DataFrame, audit: dict[str, object]) -> None:
    """Export compact evidence tables for later use in the report."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {"measure": list(audit.keys()), "value": list(audit.values())}
    ).to_csv(TABLES_DIR / "dataset_audit.csv", index=False)

    missing = (
        data.drop(columns="timestamp")
        .isna()
        .sum()
        .rename("missing_count")
        .to_frame()
    )
    missing["missing_percent"] = missing["missing_count"] / len(data) * 100
    missing.to_csv(TABLES_DIR / "missing_values.csv")

    data["cnt"].describe().rename("cnt").to_csv(
        TABLES_DIR / "target_descriptive_statistics.csv"
    )

    weather_frequency = (
        data["weathersit"].value_counts().sort_index().rename("frequency").to_frame()
    )
    weather_frequency["percent"] = weather_frequency["frequency"] / len(data) * 100
    weather_frequency.to_csv(TABLES_DIR / "weather_frequency.csv")


def export_figures(data: pd.DataFrame) -> None:
    """Export the first publication-ready EDA figure at 300 dpi."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=data,
        x="cnt",
        bins=40,
        kde=True,
        color="#2A6F97",
        ax=ax,
    )
    ax.set(
        title="Distribution of Hourly Bike Rentals",
        xlabel="Hourly rentals (cnt)",
        ylabel="Number of hours",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cnt_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_data()
    audit = audit_data(data)
    export_tables(data, audit)
    export_figures(data)

    print("UCI Bike Sharing hourly dataset audit")
    print("=" * 41)
    for measure, value in audit.items():
        print(f"{measure}: {value}")
    print(f"\nEvidence tables saved to: {TABLES_DIR}")
    print(f"Figure saved to: {FIGURES_DIR / 'cnt_distribution.png'}")


if __name__ == "__main__":
    main()
