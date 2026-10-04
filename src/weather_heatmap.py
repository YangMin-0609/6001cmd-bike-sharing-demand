from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hour.csv"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "bike_sharing_correlation_heatmap.png"

columns = ["temp", "atemp", "hum", "windspeed", "cnt"]
data = pd.read_csv(DATA_PATH)
missing = set(columns) - set(data.columns)
if missing:
    raise ValueError(f"Missing expected columns: {sorted(missing)}")

correlations = data[columns].corr(method="pearson")
labels = ["Temperature", "Feels-like temp.", "Humidity", "Wind speed", "Rentals (target)"]
correlations.index = labels
correlations.columns = labels

sns.set_theme(style="white", font_scale=1.0)
fig, ax = plt.subplots(figsize=(8.2, 6.4))
sns.heatmap(
    correlations,
    annot=False,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.7,
    linecolor="white",
    cbar_kws={"label": "Pearson correlation (r)", "shrink": 0.84},
    ax=ax,
)
for row in range(len(labels)):
    for col in range(len(labels)):
        value = correlations.iat[row, col]
        ax.text(
            col + 0.5,
            row + 0.5,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color="white" if abs(value) >= 0.75 else "#202020",
        )
ax.set_title("Correlation of weather variables and hourly rentals", pad=16, fontsize=13)
ax.tick_params(axis="x", rotation=28)
ax.tick_params(axis="y", rotation=0)
fig.text(
    0.5,
    0.015,
    "UCI Bike Sharing hour.csv (n = 17,379); identifier, category codes and target components excluded",
    ha="center",
    fontsize=8.5,
    color="#555555",
)
fig.tight_layout(rect=(0, 0.045, 1, 1))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(OUTPUT_PATH)
print(correlations.round(3).to_string())
