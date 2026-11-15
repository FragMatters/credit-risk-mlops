# MLOps POE (Proof of Experience) Comprehensive Defense Guide

## Domain & Solution Identity
- **Candidate Project**: End-to-End FinTech Credit Risk MLOps Platform
- **Application Domain**: FinTech & Banking (Retail & Commercial Lending)
- **Primary Package Manager**: `uv` (Fast Python package and project manager)
- **Focal MLOps Stack**:
  1. **MLflow** — Experiment Tracking, Hyperparameter Logging, Artifact Store, and Model Registry
  2. **pytest** — Automated Quality Gates (Data Schema, Business Invariants, Model Performance, Latency)
  3. **Evidently AI** — Continuous Data & Covariate Drift Monitoring with Interactive Visual Dashboards

---

## 1. Domain Use Case Identification (Task 1)

### Business Context & Problem Statement
In retail banking and digital consumer credit (FinTech), financial institutions extend unsecured and secured loans to retail borrowers. The primary operational risk is **Credit Default Risk** — the probability that an applicant fails to make required debt repayments within contract terms (90+ days past due / charge-off).

### Problem Objectives:
1. **Financial Objective**: Minimize credit losses (Non-Performing Assets - NPAs) while maximizing credit acceptance among creditworthy applicants.
2. **Regulatory & Risk Objective**: Ensure models adhere to Basel II/III capital adequacy standards, fair lending requirements, and algorithmic governance.
3. **MLOps Operational Objective**: Replace ad-hoc notebook-based modeling with a reproducible, test-gated, and continuously monitored pipeline.

---

## 2. Dataset & Exploratory Data Analysis (EDA) (Task 2)

### Dataset Features & Schema
The dataset contains 5,000 tabular credit applicant records (`data/credit_risk.csv`):

| Feature Column | Type | Business Description | Range / Categories |
| :--- | :--- | :--- | :--- |
| `applicant_id` | String | Unique application identifier | `APP-10000` to `APP-14999` |
| `age` | Integer | Applicant age in years | 21 – 67 years |
| `annual_income` | Float | Verified annual gross income | \$18,000 – \$300,000 |
| `loan_amount` | Float | Principal amount requested | \$2,000 – \$45,000 |
| `loan_intent` | Categorical | Purpose of the loan | `DEBTCONSOLIDATION`, `PERSONAL`, `HOMEIMPROVEMENT`, `EDUCATION`, `VENTURE`, `MEDICAL` |
| `loan_grade` | Categorical | Underwriting risk grade | `A`, `B`, `C`, `D`, `E` |
| `interest_rate` | Float | Contract interest rate | 4.5% – 26.0% |
| `credit_score` | Integer | FICO-aligned credit bureau score | 350 – 850 |
| `debt_to_income` | Float | Total debt obligations to income ratio | 0.05 – 1.80 |
| `cb_person_default_on_file` | Categorical | Prior historical credit default flag | `Y` (Yes), `N` (No) |
| `emp_length` | Integer | Years in current employment | 0 – 30 years |
| `loan_status` | Binary Target | **Target label**: 1 = Default, 0 = Non-Default | 0, 1 |

### Key EDA Insights (`reports/eda_summary.json`)
1. **Target Distribution**: Baseline default rate is **28.72%** (imbalanced classification setting reflecting real-world portfolio risk).
2. **Credit Tier Divergence**:
   - Poor credit (<580): **68.28%** default rate.
   - Fair credit (580–669): **35.87%** default rate.
   - Good credit (670–739): **20.90%** default rate.
   - Excellent credit (740+): **10.55%** default rate.
3. **Risk Grade Escalation**: Monotonically increases from Grade A (20.73%) to Grade E (51.16%).
4. **Top Correlations with Default**:
   - `credit_score` ($r = -0.3362$): Strong protective factor against default.
   - `debt_to_income` ($r = +0.3205$): Strongest risk driver of default.
   - `loan_amount` ($r = +0.2078$): Larger borrowing amounts correlate with repayment difficulty.
   - `annual_income` ($r = -0.1626$): Higher income buffers repayment risk.

### Tool Selection Finalization
While the prompt allowed selecting any one tool, our solution implements a **tri-tool synergy**:
- **MLflow** provides the backbone for model experimentation and versioned registration.
- **pytest** provides automated CI/CD gating preventing bad models or bad data from reaching staging.
- **Evidently AI** provides real-time detection of data drift and feature distribution shift post-deployment.

---

## 3. Tool Installation & Package Management (Task 3)

### Why `uv` as the Package Manager?
- **Speed**: Written in Rust, `uv` is 10–100× faster than legacy `pip` and `conda`.
- **Reproducibility**: Generates deterministic lockfiles (`uv.lock`) ensuring identical environments across developer machines and CI/CD runners.
- **Standard Standards**: Fully compliant with PEP 517, PEP 518, PEP 621 (`pyproject.toml`).

### Installation & Sync Commands
```bash
# Initialize and sync virtual environment
uv sync

# Run pipeline commands directly in isolated virtual environment
uv run python run_pipeline.py
uv run pytest -v
```

---

## 4. Problem Formulation & Technical Solution (Task 4)

### Component Architecture
```
                           +------------------------+
                           |  data/credit_risk.csv  |
                           +-----------+------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
         [ src/data/eda.py ]                     [ tests/test_data.py ]
         (Statistical EDA)                       (Schema & Quality Test)
                   |                                       |
                   v                                       v
         [ src/models/train.py ] <=================== (Gate: pytest)
         - Logistic Regression vs Random Forest
         - MLflow Tracking (Params, Metrics, Plots)
         - MLflow Model Registry (ROC-AUC >= 0.75)
                   |
                   v
         [ models/best_model.joblib ]
                   |
                   v
         [ tests/test_model.py ] (Inference Latency < 50ms & ROC-AUC > 0.75)
                   |
                   v
         [ src/monitoring/simulate_drift.py ] (Macroeconomic Shock Shift)
                   |
                   v
         [ src/monitoring/drift_report.py ]
         (Evidently AI Drift Dashboard & JSON Metrics)
```

### Detailed Tool Integration
1. **MLflow Tracking & Registry (`src/models/train.py`)**:
   - Experiment: `credit-risk-scoring` backed by SQLite store (`sqlite:///mlflow.db`).
   - Logged Parameters: test size, train/test samples, model hyperparameters.
   - Logged Metrics: ROC-AUC (0.8345), F1-Score (0.5894), Precision, Recall, LogLoss.
   - Logged Artifacts: Confusion matrix heatmaps, JSON classification reports, pipeline binaries.
   - Automated Promotion Gate: Models exceeding ROC-AUC $\ge 0.75$ are registered as versioned models in `CreditRiskClassifier`.

2. **Automated Test Suite with `pytest` (`tests/test_data.py` & `tests/test_model.py`)**:
   - 7 Data Integrity Tests: Schema column verification, zero-null assertions on mandatory fields, FICO score range bounds [350, 850], strictly positive financial checks, valid category enumerations.
   - 4 Model Quality & Latency Tests: ROC-AUC quality threshold ($\ge 0.75$), P95 inference latency SLO ($< 50$ ms), and monotonic risk directional validation (low-risk borrowers strictly yield lower default probabilities than high-risk borrowers).

3. **Production Drift Monitoring with Evidently AI (`src/monitoring/drift_report.py`)**:
   - Baseline: 4,000 reference training samples.
   - Production Batch: 2,000 live inference applicants subjected to simulated macroeconomic stress (+3.5% interest rate hike, -15% income contraction, lower credit scores).
   - Statistical Tests: Wasserstein Distance and Jensen-Shannon Divergence.
   - Detected Drift:
     - `interest_rate`: Distance 0.7201 (Statistically Significant Drift)
     - `credit_score`: Distance 0.4315 (Statistically Significant Drift)
     - `annual_income`: Distance 0.2047 (Statistically Significant Drift)
     - `debt_to_income`: Distance 0.1618 (Statistically Significant Drift)
     - `age`, `emp_length`: No drift detected (Demographics stable)
   - Outputs: Standalone interactive visual dashboard (`reports/evidently_drift_report.html`) and structured metrics (`reports/drift_metrics.json`).

---

## 5. POE External Examiner Discussion Points & Defense Script (Task 5)

When presenting to the external examiner during the POE, organize your discussion around these five pillars:

### Pillar 1: Business Context & Return on Investment (ROI)
- **Examiner Question**: *"Why is loan default prediction critical for FinTech, and how does your MLOps approach add value over traditional scoring?"*
- **Defense Response**: In banking, a 1% decrease in credit default rate translates to millions in preserved capital. Traditional credit scoring relied on static rules updated annually. Our MLOps pipeline automates continuous feature validation, compares competing models via MLflow, gates deployment with automated pytest suites, and alerts engineers via Evidently drift monitoring before portfolio losses materialize.

### Pillar 2: Architectural Tool Trade-Offs
- **Examiner Question**: *"Why did you select MLflow, pytest, and Evidently together instead of just one tool?"*
- **Defense Response**: A robust MLOps lifecycle spans three distinct phases: **Build**, **Deploy**, and **Operate**:
  1. *Build*: MLflow logs hyperparameter experiments and registers the champion model (`CreditRiskClassifier`).
  2. *Deploy*: Pytest acts as a deterministic CI/CD gate ensuring the model meets both statistical accuracy ($\text{ROC-AUC} \ge 0.75$) and operational SLAs ($\text{latency} < 50$ ms).
  3. *Operate*: Evidently AI monitors the live production environment, catching feature drift (covariate shift) before delayed credit default ground truth arrives 90 days later.

### Pillar 3: Covariate Shift vs. Concept Drift in Banking
- **Examiner Question**: *"How does your monitoring strategy handle macroeconomic shocks like interest rate hikes or recessions?"*
- **Defense Response**: In credit risk, loan repayment outcomes take 6 to 12 months to mature (**delayed ground truth**). Therefore, monitoring model accuracy alone is insufficient. By using Evidently to monitor **input feature distributions (covariate shift)**—such as interest rate increases or credit score contraction—we detect distribution shifts in real-time on day 1 of inference, triggering proactive retraining workflows.

### Pillar 4: Testing ML Pipelines Beyond Traditional Unit Tests
- **Examiner Question**: *"How does your pytest test suite differ from standard software engineering unit testing?"*
- **Defense Response**: Standard software tests check deterministic function outputs. In MLOps, we test:
  1. *Data Schemas*: Null-tolerance, valid categoricals, FICO boundaries.
  2. *Model Quality Gates*: Verifying test set ROC-AUC meets minimum thresholds.
  3. *Operational SLOs*: P95 inference latency under 50 ms.
  4. *Domain Invariants*: Directional monotonicity assertions (e.g. higher income and lower debt must yield lower default probability).

### Pillar 5: Production Deployment & Retraining Strategy
- **Examiner Question**: *"What happens when Evidently flags that 4 out of 10 features have drifted?"*
- **Defense Response**:
  1. Automated Alert: The Evidently drift report flags drift when the drifted columns share exceeds 40%.
  2. Human-in-the-Loop Review: Risk underwriters and data scientists inspect `reports/evidently_drift_report.html` to confirm whether shift is structural (e.g., central bank rate hike).
  3. Triggered Retraining: A trigger launches `src/models/train.py` on the recent windowed dataset, logs new candidate runs to MLflow, executes `pytest` gates, and creates Version 2 in the MLflow Model Registry upon approval.

---

## 6. Execution Command Reference

```bash
# 1. Generate clean dataset
uv run python src/data/make_dataset.py

# 2. Run Exploratory Data Analysis
uv run python src/data/eda.py

# 3. Train models and track in MLflow
uv run python src/models/train.py

# 4. Execute automated data and model test suites
uv run pytest -v

# 5. Simulate production drift batch
uv run python src/monitoring/simulate_drift.py

# 6. Generate Evidently AI drift reports
uv run python src/monitoring/drift_report.py

# 7. Unified single-command pipeline run
uv run python run_pipeline.py
```
