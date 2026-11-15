# FinTech Credit Risk MLOps Platform

An end-to-end, production-grade MLOps system for **Loan Default Prediction & Credit Risk Scoring**, built with **`uv`**, **MLflow**, **pytest**, and **Evidently AI**.

---

## Architecture Overview

```
                                +-------------------------+
                                |  Credit Risk Applicant  |
                                |     data/credit_risk    |
                                +------------+------------+
                                             |
                  +--------------------------+--------------------------+
                  |                                                     |
                  v                                                     v
        [ src/data/eda.py ]                                   [ tests/test_data.py ]
        - Distribution Analysis                               - Schema Verification
        - Imbalance & Correlation                             - FICO Bounds [350, 850]
        - Risk Segment Default Rates                          - Null Value Checks
                  |                                                     |
                  v                                                     v
        [ src/models/train.py ] <============================ (Quality Gate: pytest)
        - Logistic Regression & Random Forest
        - MLflow Experiment Tracking
        - MLflow Model Registry (ROC-AUC >= 0.75)
                  |
                  v
        [ models/best_model.joblib ]
                  |
                  v
        [ tests/test_model.py ] (Inference Latency < 50ms & ROC-AUC >= 0.75)
                  |
                  v
        [ src/monitoring/simulate_drift.py ] (Macroeconomic Covariate Shift)
                  |
                  v
        [ src/monitoring/drift_report.py ]
        - Evidently AI Drift Dashboard
        - Interactive HTML Report (reports/evidently_drift_report.html)
        - Machine-readable JSON Metrics (reports/drift_metrics.json)
```

---

## Tool Stack

| Tool | Role in Platform |
| :--- | :--- |
| **`uv`** | Fast Python package and dependency manager (`pyproject.toml`, `uv.lock`) |
| **MLflow** | Experiment tracking (hyperparameters, metrics, confusion matrices) & Model Registry |
| **pytest** | Automated quality gates: data integrity, model accuracy thresholds, inference latency SLOs |
| **Evidently AI** | Covariate shift and data drift detection with interactive HTML reporting |
| **Scikit-Learn** | Data preprocessing (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) and classification |

---

## Quickstart

### 1. Environment Setup with `uv`
```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Install dependencies into isolated virtual environment
uv sync
```

### 2. Run the Full End-to-End Pipeline
```bash
uv run python run_pipeline.py
```

### 3. Run Pipeline Components Individually
```bash
# 1. Generate clean dataset
uv run python src/data/make_dataset.py

# 2. Run Exploratory Data Analysis (EDA)
uv run python src/data/eda.py

# 3. Train models and track runs in MLflow
uv run python src/models/train.py --gate 0.75

# 4. Run automated test suites (data + model)
uv run pytest -v

# 5. Simulate production batch under macroeconomic stress
uv run python src/monitoring/simulate_drift.py

# 6. Generate Evidently AI drift reports
uv run python src/monitoring/drift_report.py
```

### 4. Launch MLflow Dashboard
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://localhost:5000` to inspect experiment runs, metrics, confusion matrix plots, and registered models.

---

## Jenkins CI/CD Setup

This repository includes a declarative [`Jenkinsfile`](./Jenkinsfile) ready for continuous integration and automated MLOps gating.

### Steps to Configure in Jenkins:
1. **Create Jenkins Pipeline Job**:
   - In Jenkins Dashboard, click **New Item** -> Name it `credit-risk-mlops` -> Select **Pipeline** -> Click **OK**.
2. **Configure Pipeline from SCM**:
   - Scroll down to the **Pipeline** section.
   - Set **Definition** to `Pipeline script from SCM`.
   - Set **SCM** to `Git`.
   - Enter your repository URL (e.g. `https://github.com/<username>/<repo-name>.git`).
   - Add your Git credentials (SSH Key or GitHub Personal Access Token).
   - Set **Branch Specifier** to `*/main` (or `*/master`).
   - Ensure **Script Path** is `Jenkinsfile`.
3. **Build Triggers (Optional)**:
   - Check **GitHub hook trigger for GITScm polling** or configure a webhook in your GitHub repository (`Settings -> Webhooks -> Add Webhook -> URL: http://<jenkins-url>/github-webhook/`).
4. **Run the Build**:
   - Click **Build Now**.
   - Jenkins will pull the code, install `uv`, validate schemas, train the model, assert pytest gates, generate the Evidently drift report, and archive HTML/XML reports as build artifacts.

---

## Detailed Defense Guide
For the oral exam (POE) presentation script, domain justification, and examiner Q&A, refer to [`docs/POE_EXAMINER_GUIDE.md`](./docs/POE_EXAMINER_GUIDE.md).
