"""FinTech Credit Risk dataset generator for loan default prediction.

Generates realistic tabular applicant records with demographic, financial,
and credit history attributes, accompanied by a calibrated default probability.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_credit_risk_data(
    n_samples: int = 5000,
    seed: int = 42,
    macro_shock_factor: float = 0.0,
) -> pd.DataFrame:
    """Generate synthetic credit risk applicant data.

    Parameters
    ----------
    n_samples : int
        Number of applicant records to generate.
    seed : int
        Random seed for reproducibility.
    macro_shock_factor : float
        Shift factor simulating macroeconomic pressure (higher interest, lower credit score).

    Returns
    -------
    pd.DataFrame
        Synthetic credit risk dataset with target column 'loan_status'.
    """
    rng = np.random.default_rng(seed)

    applicant_ids = [f"APP-{10000 + i}" for i in range(n_samples)]
    age = rng.integers(21, 68, size=n_samples)
    emp_length = np.clip(
        np.round(rng.exponential(scale=5.0, size=n_samples)), 0, age - 18
    )

    # Base income log-normal
    annual_income = np.round(
        rng.lognormal(mean=10.9, sigma=0.55, size=n_samples)
        * (1.0 - 0.15 * macro_shock_factor),
        -2,
    )
    annual_income = np.clip(annual_income, 18000, 300000)

    # Loan amounts
    loan_amount = np.round(
        rng.triangular(left=2000, mode=12000, right=45000, size=n_samples), -2
    )

    loan_intents = [
        "DEBTCONSOLIDATION",
        "PERSONAL",
        "HOMEIMPROVEMENT",
        "EDUCATION",
        "VENTURE",
        "MEDICAL",
    ]
    intent_probs = [0.35, 0.20, 0.15, 0.12, 0.10, 0.08]
    loan_intent = rng.choice(loan_intents, size=n_samples, p=intent_probs)

    loan_grades = ["A", "B", "C", "D", "E"]
    grade_probs = [0.25, 0.30, 0.25, 0.14, 0.06]
    loan_grade = rng.choice(loan_grades, size=n_samples, p=grade_probs)

    # Credit scores: 350 - 850
    base_credit_score = rng.normal(loc=675, scale=65, size=n_samples) - (
        30.0 * macro_shock_factor
    )
    credit_score = np.clip(np.round(base_credit_score), 350, 850).astype(int)

    # Interest rate depends on loan grade and macro factor
    grade_rate_map = {"A": 7.5, "B": 11.0, "C": 14.5, "D": 18.0, "E": 21.5}
    base_rates = np.array([grade_rate_map[g] for g in loan_grade])
    interest_rate = np.round(
        base_rates
        + rng.normal(0, 1.2, size=n_samples)
        + (3.5 * macro_shock_factor),
        2,
    )
    interest_rate = np.clip(interest_rate, 4.5, 26.0)

    # Debt-to-income ratio (DTI)
    debt_to_income = np.round((loan_amount / annual_income) + rng.uniform(0.05, 0.20, size=n_samples), 4)

    # Historical default flag (Y / N)
    prior_default_prob = np.where(credit_score < 600, 0.45, 0.12)
    cb_person_default_on_file = np.where(
        rng.random(size=n_samples) < prior_default_prob, "Y", "N"
    )

    # Logistic default risk probability computation
    prior_default_num = np.where(cb_person_default_on_file == "Y", 1.0, 0.0)
    z = (
        -2.5
        + 3.0 * debt_to_income
        - 0.014 * (credit_score - 620)
        + 0.14 * (interest_rate - 10)
        + 1.4 * prior_default_num
        - 0.05 * emp_length
    )
    prob_default = 1.0 / (1.0 + np.exp(-z))
    loan_status = (rng.random(size=n_samples) < prob_default).astype(int)

    df = pd.DataFrame(
        {
            "applicant_id": applicant_ids,
            "age": age,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "interest_rate": interest_rate,
            "credit_score": credit_score,
            "debt_to_income": debt_to_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "emp_length": emp_length.astype(int),
            "loan_status": loan_status,
        }
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic credit risk dataset.")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of records to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/credit_risk.csv", help="Output CSV path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_credit_risk_data(n_samples=args.n_samples, seed=args.seed)
    df.to_csv(out_path, index=False)
    print(
        f"Generated {len(df)} records saved to {out_path} (Default rate: {df['loan_status'].mean():.2%})"
    )


if __name__ == "__main__":
    main()
