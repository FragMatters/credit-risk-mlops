"""Automated Exploratory Data Analysis (EDA) module for FinTech credit risk dataset.

Computes summary statistics, missing value profiles, category distributions,
risk correlations, and target default rates by segment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def perform_eda(df: pd.DataFrame) -> dict:
    """Analyze credit risk dataset and return comprehensive EDA metrics."""
    # 1. Basic Metadata
    total_records = int(len(df))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 2. Missing Value Analysis
    missing_counts = df.isnull().sum().to_dict()
    missing_percentages = (df.isnull().mean() * 100.0).round(2).to_dict()

    # 3. Target Distribution
    target_counts = df["loan_status"].value_counts().to_dict()
    default_rate = float(df["loan_status"].mean())

    # 4. Numerical Summary Statistics
    numeric_stats = {}
    for col in numeric_cols:
        series = df[col]
        numeric_stats[col] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "p25": round(float(series.quantile(0.25)), 4),
            "median": round(float(series.median()), 4),
            "p75": round(float(series.quantile(0.75)), 4),
            "max": round(float(series.max()), 4),
        }

    # 5. Default Rate by Risk Segments
    default_by_grade = (
        df.groupby("loan_grade")["loan_status"].mean().round(4).to_dict()
    )
    default_by_intent = (
        df.groupby("loan_intent")["loan_status"].mean().round(4).to_dict()
    )
    default_by_prior_default = (
        df.groupby("cb_person_default_on_file")["loan_status"].mean().round(4).to_dict()
    )

    # Credit score tiers
    bins = [300, 580, 670, 740, 850]
    labels = ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Excellent (740+)"]
    score_tier = pd.cut(df["credit_score"], bins=bins, labels=labels)
    default_by_score_tier = (
        df.groupby(score_tier, observed=False)["loan_status"].mean().round(4).to_dict()
    )
    default_by_score_tier = {str(k): float(v) for k, v in default_by_score_tier.items()}

    # 6. Feature Correlation with Target
    correlations = {}
    for col in numeric_cols:
        if col != "loan_status":
            corr = float(df[col].corr(df["loan_status"]))
            correlations[col] = round(corr, 4)

    # Sort correlations descending by absolute impact
    sorted_correlations = dict(
        sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
    )

    eda_summary = {
        "dataset_metadata": {
            "total_records": total_records,
            "total_features": len(df.columns),
            "numerical_features": numeric_cols,
            "categorical_features": categorical_cols,
        },
        "missing_values": {
            "counts": missing_counts,
            "percentages": missing_percentages,
        },
        "target_distribution": {
            "counts": {str(k): int(v) for k, v in target_counts.items()},
            "default_rate": round(default_rate, 4),
            "default_percentage": f"{default_rate * 100.0:.2f}%",
        },
        "numerical_statistics": numeric_stats,
        "risk_segment_default_rates": {
            "by_loan_grade": default_by_grade,
            "by_loan_intent": default_by_intent,
            "by_prior_default": default_by_prior_default,
            "by_credit_score_tier": default_by_score_tier,
        },
        "feature_correlations_with_default": sorted_correlations,
    }

    return eda_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on credit risk dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/credit_risk.csv",
        help="Path to credit risk CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/eda_summary.json",
        help="Output path for EDA summary JSON",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input dataset not found at {in_path}")

    df = pd.read_csv(in_path)
    summary = perform_eda(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("FINTECH CREDIT RISK: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"Total applicants: {summary['dataset_metadata']['total_records']}")
    print(f"Baseline default rate: {summary['target_distribution']['default_percentage']}")
    print("\nDefault Rate by Credit Tier:")
    for tier, rate in summary["risk_segment_default_rates"]["by_credit_score_tier"].items():
        print(f"  {tier:20}: {rate:.2%}")
    print("\nDefault Rate by Loan Grade:")
    for grade, rate in summary["risk_segment_default_rates"]["by_loan_grade"].items():
        print(f"  Grade {grade}: {rate:.2%}")
    print("\nTop Correlations with Default Risk:")
    for feat, corr in list(summary["feature_correlations_with_default"].items())[:5]:
        print(f"  {feat:20}: {corr:+.4f}")
    print(f"\nFull EDA summary saved to: {out_path}")


if __name__ == "__main__":
    main()
