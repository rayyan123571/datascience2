"""
Generate the full EDA visualisation set for the loan-prediction dataset.

Writes ~16 PNG files to ../eda_images and a one-page text summary to
../reports/eda_summary.txt.

Usage:  python eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data" / "Bank_Personal_Loan_Modelling.csv"
IMG_DIR = ROOT / "eda_images"
REPORT_DIR = ROOT / "reports"

TARGET = "Personal Loan"
NUMERIC = ["Age", "Experience", "Income", "CCAvg", "Mortgage"]
CATEGORICAL = ["Family", "Education", "Securities Account", "CD Account",
               "Online", "CreditCard"]

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150})


def _save(fig, name: str) -> None:
    out = IMG_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {out.relative_to(ROOT)}")


def class_balance(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["No Loan (0)", "Loan (1)"], counts.values,
                  color=["#4F8FFF", "#FF5252"])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 30, f"{v}\n({v/len(df):.1%})",
                ha="center", fontweight="bold")
    ax.set_title("Target Class Balance — Personal Loan", fontweight="bold")
    ax.set_ylabel("Count")
    _save(fig, "01_class_balance.png")


def numeric_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold")
    for ax, col in zip(axes.flat, NUMERIC):
        sns.histplot(df[col], bins=30, kde=True, ax=ax, color="#4F8FFF")
        ax.set_title(col, fontweight="bold")
    axes.flat[-1].axis("off")
    _save(fig, "02_numeric_distributions.png")


def numeric_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Numerical Feature Boxplots (outlier check)", fontsize=14,
                 fontweight="bold")
    for ax, col in zip(axes.flat, NUMERIC):
        sns.boxplot(x=df[col], ax=ax, color="#FFB800")
        ax.set_title(col, fontweight="bold")
    axes.flat[-1].axis("off")
    _save(fig, "03_numeric_boxplots.png")


def feature_vs_target_box(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Numeric Features by Loan Acceptance", fontsize=14,
                 fontweight="bold")
    for ax, col in zip(axes.flat, NUMERIC):
        sns.boxplot(x=TARGET, y=col, data=df, ax=ax, hue=TARGET,
                    palette={0: "#4F8FFF", 1: "#FF5252"}, legend=False)
        ax.set_title(col, fontweight="bold")
        ax.set_xticklabels(["No", "Yes"])
    axes.flat[-1].axis("off")
    _save(fig, "04_features_by_target.png")


def correlation_heatmap(df: pd.DataFrame) -> None:
    drop = ["ID", "ZIP Code"]
    sub = df.drop(columns=[c for c in drop if c in df.columns])
    corr = sub.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation Heatmap", fontweight="bold")
    _save(fig, "05_correlation_heatmap.png")


def acceptance_by_categorical(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Loan Acceptance Rate by Categorical Features", fontsize=14,
                 fontweight="bold")
    for ax, col in zip(axes.flat, CATEGORICAL):
        rate = df.groupby(col)[TARGET].mean()
        sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax,
                    color="#00D084")
        ax.set_title(col, fontweight="bold")
        ax.set_ylabel("Acceptance Rate")
        ax.set_ylim(0, max(0.5, rate.max() * 1.2))
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")
    _save(fig, "06_acceptance_by_categorical.png")


def income_vs_ccavg_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df, x="Income", y="CCAvg", hue=TARGET,
                    palette={0: "#4F8FFF", 1: "#FF5252"}, alpha=0.55,
                    ax=ax, s=35)
    ax.set_title("Income vs Credit-Card Average — coloured by loan outcome",
                 fontweight="bold")
    ax.set_xlabel("Income ($1000s)")
    ax.set_ylabel("CC Avg ($1000s)")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Personal Loan")
    _save(fig, "07_income_vs_ccavg.png")


def income_distribution_by_target(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in [(0, "#4F8FFF"), (1, "#FF5252")]:
        sub = df[df[TARGET] == label]["Income"]
        ax.hist(sub, bins=40, alpha=0.55, color=color,
                label=("No Loan" if label == 0 else "Loan"))
    ax.set_title("Income Distribution by Loan Outcome", fontweight="bold")
    ax.set_xlabel("Income ($1000s)")
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, "08_income_by_outcome.png")


def age_distribution_by_target(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in [(0, "#4F8FFF"), (1, "#FF5252")]:
        sub = df[df[TARGET] == label]["Age"]
        ax.hist(sub, bins=30, alpha=0.55, color=color,
                label=("No Loan" if label == 0 else "Loan"))
    ax.set_title("Age Distribution by Loan Outcome", fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, "09_age_by_outcome.png")


def pairplot(df: pd.DataFrame) -> None:
    cols = ["Income", "CCAvg", "Mortgage", "Experience", TARGET]
    sub = df[cols].copy()
    sub[TARGET] = sub[TARGET].map({0: "No Loan", 1: "Loan"})
    g = sns.pairplot(sub, hue=TARGET, palette={"No Loan": "#4F8FFF",
                                                "Loan": "#FF5252"},
                     plot_kws={"alpha": 0.5, "s": 14}, diag_kind="kde",
                     height=2.4)
    g.fig.suptitle("Pairplot of Top Numeric Features", y=1.01,
                   fontweight="bold")
    g.fig.savefig(IMG_DIR / "10_pairplot.png", bbox_inches="tight", dpi=120)
    plt.close(g.fig)
    print(f"  saved  eda_images/10_pairplot.png")


def education_income_violin(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="Education", y="Income", hue=TARGET, data=df,
                   split=True, palette={0: "#4F8FFF", 1: "#FF5252"}, ax=ax)
    ax.set_title("Income Distribution by Education and Loan Outcome",
                 fontweight="bold")
    ax.set_xticklabels(["Undergrad", "Graduate", "Advanced"])
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Personal Loan")
    _save(fig, "11_education_income_violin.png")


def family_acceptance_stack(df: pd.DataFrame) -> None:
    ct = pd.crosstab(df["Family"], df[TARGET], normalize="index")
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=["#4F8FFF", "#FF5252"], width=0.7)
    ax.set_title("Loan Outcome Share by Family Size", fontweight="bold")
    ax.set_ylabel("Share")
    ax.set_xlabel("Family Size")
    ax.legend(["No Loan", "Loan"], title="Personal Loan")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    _save(fig, "12_family_stack.png")


def outlier_summary(df: pd.DataFrame) -> dict:
    summary = {}
    for col in NUMERIC:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        summary[col] = {"n_outliers": n_out,
                        "pct": n_out / len(df) * 100,
                        "lower": float(lo), "upper": float(hi)}
    return summary


def write_summary(df: pd.DataFrame, outliers: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "eda_summary.txt"
    lines = []
    lines.append("=" * 70)
    lines.append("EXPLORATORY DATA ANALYSIS — SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Records             : {len(df):,}")
    lines.append(f"Features            : {df.shape[1]}")
    lines.append(f"Missing values      : {int(df.isna().sum().sum())}")
    lines.append(f"Duplicate rows      : {int(df.duplicated().sum())}")
    lines.append(f"Target acceptance   : {df[TARGET].mean():.2%}")
    lines.append("")
    lines.append("Class distribution:")
    for cls, n in df[TARGET].value_counts().sort_index().items():
        lines.append(f"  Class {cls}: {n:>5,}  ({n/len(df):.2%})")

    lines.append("")
    lines.append("Numeric feature summary:")
    desc = df[NUMERIC].describe().round(2).T
    lines.append(desc.to_string())

    lines.append("")
    lines.append("Outliers (IQR method):")
    for col, info in outliers.items():
        lines.append(f"  {col:10s}  {info['n_outliers']:>4d} rows "
                     f"({info['pct']:.2f}%)  range [{info['lower']:.2f}, {info['upper']:.2f}]")

    lines.append("")
    lines.append("Top correlations with target:")
    drop = ["ID", "ZIP Code"]
    corr = (df.drop(columns=[c for c in drop if c in df.columns])
            .corr(numeric_only=True)[TARGET]
            .drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False))
    for k, v in corr.items():
        lines.append(f"  {k:25s}  {v:+.3f}")

    out.write_text("\n".join(lines))
    print(f"  saved  {out.relative_to(ROOT)}")


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"Dataset not found at {DATA}")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} rows from {DATA.name}")
    print(f"Writing plots to {IMG_DIR.relative_to(ROOT)}/\n")

    class_balance(df)
    numeric_distributions(df)
    numeric_boxplots(df)
    feature_vs_target_box(df)
    correlation_heatmap(df)
    acceptance_by_categorical(df)
    income_vs_ccavg_scatter(df)
    income_distribution_by_target(df)
    age_distribution_by_target(df)
    pairplot(df)
    education_income_violin(df)
    family_acceptance_stack(df)

    outliers = outlier_summary(df)
    write_summary(df, outliers)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
