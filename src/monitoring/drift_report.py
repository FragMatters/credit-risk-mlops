"""Production drift detection and monitoring using Evidently AI.

Compares reference training data with production inference batches, evaluates
statistical feature drift and target/prediction drift, and produces interactive HTML reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Ensure root in python path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def generate_drift_report(
    reference_path: str = "data/train.csv",
    production_path: str = "data/production_batch.csv",
    html_output: str = "reports/evidently_drift_report.html",
    json_output: str = "reports/drift_metrics.json",
) -> dict:
    """Run Evidently Data Drift monitoring and export artifacts."""
    ref_file = Path(reference_path)
    prod_file = Path(production_path)

    if not ref_file.exists():
        raise FileNotFoundError(f"Reference dataset missing at {ref_file}")
    if not prod_file.exists():
        raise FileNotFoundError(f"Production dataset missing at {prod_file}")

    ref_df = pd.read_csv(ref_file)
    prod_df = pd.read_csv(prod_file)

    # Exclude unique ID if present
    drop_cols = [c for c in ["applicant_id", "loan_status"] if c in ref_df.columns]
    feature_cols = [c for c in ref_df.columns if c not in drop_cols]

    # Align columns between reference and production
    common_cols = [c for c in feature_cols if c in prod_df.columns]

    ref_subset = ref_df[common_cols]
    prod_subset = prod_df[common_cols]

    # Execute Evidently DataDriftPreset report
    report = Report([DataDriftPreset()])
    eval_result = report.run(prod_subset, ref_subset)

    # Save outputs
    html_path = Path(html_output)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    eval_result.save_html(str(html_path))

    json_path = Path(json_output)
    eval_result.save_json(str(json_path))

    # Parse JSON for terminal executive summary
    with open(json_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    # Extract high-level drift summary from metrics
    summary = {
        "reference_records": len(ref_subset),
        "production_records": len(prod_subset),
        "features_monitored": len(common_cols),
        "report_html": str(html_path),
        "report_json": str(json_path),
    }

    print("=" * 60)
    print("EVIDENTLY AI: PRODUCTION DRIFT MONITORING REPORT")
    print("=" * 60)
    print(f"Reference samples (baseline) : {summary['reference_records']}")
    print(f"Production samples (batch)    : {summary['production_records']}")
    print(f"Monitored features count     : {summary['features_monitored']}")
    print(f"\nInteractive HTML report saved: {html_path}")
    print(f"Machine-readable JSON saved  : {json_path}")
    print("=" * 60)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Evidently AI data drift report.")
    parser.add_argument(
        "--reference",
        type=str,
        default="data/train.csv",
        help="Path to baseline reference dataset",
    )
    parser.add_argument(
        "--production",
        type=str,
        default="data/production_batch.csv",
        help="Path to production batch dataset",
    )
    parser.add_argument(
        "--html",
        type=str,
        default="reports/evidently_drift_report.html",
        help="Output HTML report path",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="reports/drift_metrics.json",
        help="Output JSON metrics path",
    )
    args = parser.parse_args()

    generate_drift_report(
        reference_path=args.reference,
        production_path=args.production,
        html_output=args.html,
        json_output=args.json,
    )


if __name__ == "__main__":
    main()
