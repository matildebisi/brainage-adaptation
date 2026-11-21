from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

import argparse

_parser = argparse.ArgumentParser(description='Plot signed error versus age.')
_parser.add_argument('--input', required=True, type=Path)
_parser.add_argument('--output', required=True, type=Path)
_args = _parser.parse_args()
ERRORS_CSV = _args.input
OUT_PATH = _args.output
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

AGE_COL = "Age"

CATEGORY_ORDER = [
   # "Zero-shot",
    "Traditional",
    "LoRA",
    "DoRA",
    "PiSSA",
    "FiltLoRA",
]

TECHNIQUE_COLORS = {
    "Zero-shot":   "#777777",
    "Traditional": "#2F5D8A",
    "LoRA":        "#C84C7A",
    "DoRA":        "#D69A2D",
    "PiSSA":       "#2E8B57",
    "FiltLoRA":    "#9C6BCF",
}

DPI = 300


# =============================================================================
# HELPERS
# =============================================================================

def linear_fit_with_ci(x, y, n_grid=200):
    """
    Linear regression y ~ x with 95% confidence interval
    for the estimated mean relationship.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < 3 or np.unique(x).size < 2:
        return None

    n = len(x)

    # Linear regression
    slope, intercept = np.polyfit(x, y, 1)

    x_grid = np.linspace(
        x.min(),
        x.max(),
        n_grid,
    )

    y_fit = intercept + slope * x_grid

    # Residual variance
    y_hat = intercept + slope * x
    residuals = y - y_hat

    sse = np.sum(residuals ** 2)
    dof = n - 2

    if dof <= 0:
        return None

    residual_se = np.sqrt(sse / dof)

    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)

    # SE of fitted mean
    fit_se = residual_se * np.sqrt(
        1 / n
        + (x_grid - x_mean) ** 2 / sxx
    )

    # 95% CI
    tcrit = 1.96
    ci_low = y_fit - tcrit * fit_se
    ci_high = y_fit + tcrit * fit_se

    return x_grid, y_fit, ci_low, ci_high


# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(ERRORS_CSV)

df[AGE_COL] = pd.to_numeric(
    df[AGE_COL],
    errors="coerce",
)

df["signed_error"] = pd.to_numeric(
    df["signed_error"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        AGE_COL,
        "signed_error",
        "category",
    ]
).copy()

categories = [
    c
    for c in CATEGORY_ORDER
    if c in df["Adaptation"].unique()
]


# =============================================================================
# PLOT
# =============================================================================

fig, ax = plt.subplots(
    figsize=(9.5, 6.5)
)

for category in categories:

    d = df[
        df["Adaptation"] == category
    ].copy()

    color = TECHNIQUE_COLORS[category]

    # ---------------------------------------------------------
    # Subject-level points
    # ---------------------------------------------------------
    ax.scatter(
        d[AGE_COL],
        d["signed_error"],
        s=34,
        alpha=0.42,
        color=color,
        edgecolors="none",
        zorder=2,
    )

    # ---------------------------------------------------------
    # Linear trend + 95% confidence band
    # ---------------------------------------------------------
    fit = linear_fit_with_ci(
        d[AGE_COL],
        d["signed_error"],
    )

    if fit is not None:

        x_grid, y_fit, ci_low, ci_high = fit

        ax.fill_between(
            x_grid,
            ci_low,
            ci_high,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            x_grid,
            y_fit,
            color=color,
            linewidth=2.6,
            label=category,
            zorder=3,
        )


# =============================================================================
# REFERENCE LINE
# =============================================================================

ax.axhline(
    0,
    color="#555555",
    linestyle="--",
    linewidth=1.3,
    alpha=0.8,
    zorder=0,
)


# =============================================================================
# STYLE
# =============================================================================

ax.set_xlabel(
    "Chronological age (years)",
    fontsize=16,
    fontweight="bold",
)

ax.set_ylabel(
    "Signed prediction error (years)",
    fontsize=16,
    fontweight="bold",
)

ax.tick_params(
    axis="both",
    labelsize=14,
)

ax.set_title("")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)

ax.grid(
    axis="y",
    alpha=0.15,
    linewidth=0.8,
)

ax.set_axisbelow(True)


# =============================================================================
# LEGEND
# =============================================================================

legend = ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=12,
    handlelength=2.2,
)

for text in legend.get_texts():
    text.set_fontweight("bold")


# =============================================================================
# SAVE
# =============================================================================

plt.tight_layout()

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    OUT_PATH,
    dpi=DPI,
    bbox_inches="tight",
)

plt.show()