# Proposal

## Why
Financial institutions face substantial credit risk and regulatory scrutiny when evaluating loan applications, requiring automated, reproducible, and verifiable machine learning pipelines. For this MLOps Proof of Experience (POE), we design and implement a production-grade FinTech Credit Risk assessment solution packaged with `uv` that integrates MLflow for experiment tracking and model registration, pytest for automated data validation and model quality testing, and Evidently for continuous data drift and feature distribution monitoring.

## What Changes
- Introduce a high-performance Python environment managed with `uv` configured with `pyproject.toml` containing all core ML and MLOps dependencies (`mlflow`, `pytest`, `evidently`, `scikit-learn`, `pandas`, `joblib`).
- Implement a clean FinTech Credit Risk dataset generator and Exploratory Data Analysis (EDA) module covering feature distributions, missingness, correlation analysis, and target imbalance.
- Implement an end-to-end classification pipeline for loan default prediction with feature transformation and baseline/optimized model training.
- Integrate MLflow experiment tracking to log model parameters, metrics (ROC-AUC, Precision, Recall, F1), artifacts, feature importance plots, and model registry artifacts.
- Implement an automated pytest validation suite verifying data schema integrity, feature null checks, model latency, and minimum performance thresholds.
- Implement Evidently AI monitoring pipelines that generate interactive Data Drift and Model Performance reports comparing reference training distributions with production inference batches.
- Provide an external POE discussion dossier covering architecture trade-offs, MLOps maturity dimensions, and examiner interview points.

## Capabilities

### New Capabilities
- `credit-risk-pipeline`: Ingestion, exploratory data analysis (EDA), feature preprocessing, model training, and inference for loan default risk prediction.
- `mlops-governance`: Experiment tracking and model registry with MLflow, automated data and model test suites with pytest, and data drift monitoring with Evidently.

### Modified Capabilities
None.

## Impact
- Build & Package Management: Modern `uv` toolchain for deterministic virtual environments and sub-second dependency resolution.
- Storage & Artifacts: Local MLflow tracking store (`mlruns/`), serialized model pipelines, and Evidently visual HTML reports.
- Codebase: Modular architecture separating data generation, EDA, training, testing, and monitoring components.
