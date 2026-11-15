"""Unified End-to-End MLOps Pipeline Orchestrator.

Executes the complete Credit Risk MLOps lifecycle:
1. Synthetic Data Generation (FinTech Banking domain)
2. Automated Exploratory Data Analysis (EDA)
3. Model Training, MLflow Tracking & Model Registry
4. Automated pytest Validation Suite (Data, Model, Latency)
5. Drift Simulation (Macroeconomic covariate shift)
6. Evidently AI Drift Report Generation (Interactive HTML & JSON)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import pytest

# Ensure repository root in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Suppress warnings and configure writable directories
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from src.data.eda import perform_eda
from src.data.make_dataset import generate_credit_risk_data
from src.models.train import train_and_evaluate
from src.monitoring.drift_report import generate_drift_report
from src.monitoring.simulate_drift import generate_production_batch


def main() -> None:
    print("\n" + "#" * 70)
    print("STARTING END-TO-END FINTECH CREDIT RISK MLOPS PIPELINE")
    print("#" * 70)

    # Step 1: Data Generation
    print("\n>>> STEP 1: Generating FinTech Credit Risk Dataset...")
    df_raw = generate_credit_risk_data(n_samples=5000, seed=42)
    data_path = Path("data/credit_risk.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(data_path, index=False)
    print(f"  [OK] Saved {len(df_raw)} records to {data_path} (Default Rate: {df_raw['loan_status'].mean():.2%})")

    # Step 2: Automated EDA
    print("\n>>> STEP 2: Running Automated Exploratory Data Analysis (EDA)...")
    eda_summary = perform_eda(df_raw)
    eda_out = Path("reports/eda_summary.json")
    eda_out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(eda_out, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"  [OK] Saved EDA summary to {eda_out}")
    print(f"  [INFO] Poor Credit Default: {eda_summary['risk_segment_default_rates']['by_credit_score_tier']['Poor (<580)']:.2%}")
    print(f"  [INFO] Grade E Default: {eda_summary['risk_segment_default_rates']['by_loan_grade']['E']:.2%}")

    # Step 3: Model Training with MLflow Tracking & Registry
    print("\n>>> STEP 3: Training Models with MLflow Tracking & Registry...")
    results = train_and_evaluate(
        data_path=str(data_path),
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="credit-risk-scoring",
        min_roc_auc_gate=0.75,
    )
    for model_name, info in results.items():
        if model_name != "registered_model":
            print(f"  [MODEL] {model_name:20} -> ROC-AUC: {info['metrics']['roc_auc']:.4f} (Gate Passed: {info['passed_gate']})")
    if "registered_model" in results:
        reg = results["registered_model"]
        print(f"  [REGISTRY] Registered '{reg['name']}' (Version {reg['version']}) with ROC-AUC {reg['best_roc_auc']:.4f}")

    # Step 4: Automated Testing Suite via pytest
    print("\n>>> STEP 4: Executing Automated Testing Gates with pytest...")
    ret_code = pytest.main(["-v", "tests"])
    if ret_code != 0:
        raise RuntimeError(f"Pytest suite failed with return code {ret_code}")
    print("  [OK] All pytest test suites passed successfully!")

    # Step 5: Simulate Production Drift Batch
    print("\n>>> STEP 5: Simulating Production Inference Batch with Macroeconomic Drift...")
    prod_path = Path("data/production_batch.csv")
    df_prod = generate_production_batch(
        n_samples=2000,
        seed=999,
        macro_shock_factor=0.85,
        output_path=str(prod_path),
    )
    print(f"  [OK] Generated {len(df_prod)} production records with simulated macroeconomic drift")

    # Step 6: Evidently AI Drift Monitoring Report
    print("\n>>> STEP 6: Generating Evidently AI Data Drift Report...")
    drift_summary = generate_drift_report(
        reference_path="data/train.csv",
        production_path=str(prod_path),
        html_output="reports/evidently_drift_report.html",
        json_output="reports/drift_metrics.json",
    )
    print(f"  [OK] Interactive HTML dashboard: {drift_summary['report_html']}")
    print(f"  [OK] Machine-readable metrics:  {drift_summary['report_json']}")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("All ML, MLOps, testing, and monitoring components verified.")
    print("Refer to docs/POE_EXAMINER_GUIDE.md for the oral defense script.\n")


if __name__ == "__main__":
    main()
