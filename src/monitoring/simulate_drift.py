"""Simulates a production inference batch with macroeconomic drift.

Simulates covariate shift (higher interest rates, lowered credit scores,
elevated debt-to-income) reflecting economic stress, and attaches model predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
from src.data.make_dataset import generate_credit_risk_data

MODEL_PATH = Path("models/best_model.joblib")


def generate_production_batch(
    n_samples: int = 2000,
    seed: int = 999,
    macro_shock_factor: float = 0.85,
    output_path: str = "data/production_batch.csv",
) -> pd.DataFrame:
    """Generate production inference batch with simulated covariate and prediction drift."""
    df = generate_credit_risk_data(
        n_samples=n_samples, seed=seed, macro_shock_factor=macro_shock_factor
    )

    # Attach model inference predictions if model exists
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        feature_cols = [
            "age",
            "annual_income",
            "loan_amount",
            "loan_intent",
            "loan_grade",
            "interest_rate",
            "credit_score",
            "debt_to_income",
            "cb_person_default_on_file",
            "emp_length",
        ]
        df["prediction"] = model.predict(df[feature_cols])
        df["prediction_score"] = model.predict_proba(df[feature_cols])[:, 1]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(
        f"Simulated production batch generated: {len(df)} records saved to {out_file} (Macro shock: {macro_shock_factor})"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate drifted production data.")
    parser.add_argument("--n-samples", type=int, default=2000, help="Batch size")
    parser.add_argument("--seed", type=int, default=999, help="Random seed")
    parser.add_argument(
        "--macro-shock", type=float, default=0.85, help="Macroeconomic drift intensity"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/production_batch.csv",
        help="Output CSV destination",
    )
    args = parser.parse_args()

    generate_production_batch(
        n_samples=args.n_samples,
        seed=args.seed,
        macro_shock_factor=args.macro_shock,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
