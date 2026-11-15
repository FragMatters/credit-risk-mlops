# Tasks

## 1. Project Initialization and Dependency Management with uv

- [x] 1.1 Initialize the Python project and configure `pyproject.toml` with dependencies (`mlflow`, `pytest`, `evidently`, `scikit-learn`, `pandas`, `numpy`, `joblib`), verifying environment creation with `uv sync`.
- [x] 1.2 Scaffold modular directory structure (`src/data`, `src/models`, `src/monitoring`, `tests`, `reports`, `docs`) and verify package layout.

## 2. Dataset Generation and Exploratory Data Analysis (EDA)

- [x] 2.1 Implement reproducible FinTech credit risk dataset generator in `src/data/make_dataset.py` with demographic, loan, and financial features, verifying CSV generation in `data/credit_risk.csv`.
- [x] 2.2 Implement automated EDA module in `src/data/eda.py` computing summary statistics, missingness, feature correlations, and target default distribution, verifying generation of `reports/eda_summary.json`.

## 3. Credit Risk Model Training & MLflow Governance

- [x] 3.1 Implement ML training pipeline in `src/models/train.py` fitting Logistic Regression and Random Forest classifiers, logging hyperparameters, ROC-AUC, Precision, Recall, F1 score, and confusion matrix artifacts to MLflow, verifying logged runs in `mlruns/`.
- [x] 3.2 Implement model evaluation gate and MLflow Model Registry registration for models meeting minimum ROC-AUC threshold (>= 0.75), verifying registered model artifact creation.

## 4. Automated Testing Suite with pytest

- [x] 4.1 Implement data integrity and schema validation tests in `tests/test_data.py` asserting non-null constraints, feature ranges, and schema consistency, verifying test execution with `uv run pytest tests/test_data.py`.
- [x] 4.2 Implement model quality and inference latency gate tests in `tests/test_model.py` verifying ROC-AUC > 0.75 and inference latency < 50ms, verifying test suite passes with `uv run pytest`.

## 5. Drift Monitoring with Evidently AI

- [x] 5.1 Implement production inference batch generator with simulated macroeconomic drift in `src/monitoring/simulate_drift.py`, verifying drifted batch generation in `data/production_batch.csv`.
- [x] 5.2 Implement Evidently AI drift detection pipeline in `src/monitoring/drift_report.py` comparing reference training data against production batch, verifying generation of interactive `reports/evidently_drift_report.html` and `reports/drift_metrics.json`.

## 6. POE External Examiner Guide and End-to-End Verification

- [x] 6.1 Create `docs/POE_EXAMINER_GUIDE.md` detailing domain problem formulation, EDA insights, tool trade-offs (MLflow, pytest, Evidently), and oral defense discussion questions, verifying documentation completeness.
- [x] 6.2 Implement unified workflow orchestrator script `run_pipeline.py` executing EDA, model training, test suites, and drift report generation, verifying end-to-end execution without errors.
