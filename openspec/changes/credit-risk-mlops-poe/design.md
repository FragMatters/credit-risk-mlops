# Design

## Context
The project addresses credit default risk prediction within the FinTech and banking domain. Financial institutions require strict model transparency, reproducibility, and continuous monitoring to comply with financial regulations and minimize bad loan exposure. Currently, there is no existing code in the repository. We design a modular, production-ready MLOps solution using `uv` for lightning-fast package management, integrating MLflow for experiment tracking and model registry, pytest for automated data and model quality testing, and Evidently for continuous drift detection. See `proposal.md` for motivation and background.

## Goals / Non-Goals

**Goals:**
- Provide a clean, reproducible FinTech credit risk dataset with realistic banking features (age, annual income, loan amount, credit score, debt-to-income ratio, number of open credit lines, default target).
- Implement an automated Exploratory Data Analysis (EDA) module exporting summary statistics, missing value profiles, and risk correlation matrices.
- Develop a machine learning classification pipeline (e.g., Logistic Regression and Random Forest/Gradient Boosting) predicting borrower default probability.
- Instrument the pipeline with MLflow to track parameters, evaluation metrics (ROC-AUC, Precision, Recall, F1, LogLoss), artifacts (confusion matrices, feature importance plots), and model registry versioning.
- Implement a comprehensive pytest validation suite acting as an MLOps deployment gate (verifying schema conformity, null tolerances, minimum ROC-AUC > 0.75, and inference latency < 50ms).
- Integrate Evidently AI to detect covariate shift and target drift between baseline training data and production inference batches, outputting interactive HTML dashboards.
- Deliver a dedicated POE External Examiner Guide detailing architecture trade-offs, operational mechanics, and defense talking points.

**Non-Goals:**
- Deploying to managed cloud Kubernetes or AWS SageMaker endpoints (local execution with CLI entry points is scoped for POE evaluation).
- Complex streaming ingestion with Kafka or real-time event brokers (batch-oriented processing and inference are prioritized).

## Decisions

### Decision 1: Dependency Management with `uv`
- **Choice**: Use Astral's `uv` toolchain (`uv init`, `pyproject.toml`, `uv run`) for environment and dependency management.
- **Rationale**: `uv` provides 10-100x faster dependency resolution than traditional `pip` or `conda`, deterministic lockfile support, and virtual environment isolation without system-level pollution.
- **Alternatives Considered**: Standard `pip` with `requirements.txt` (slower, prone to unpinned dependency drift); `conda`/`poetry` (heavier overhead, slower environment solves).

### Decision 2: MLflow for Tracking and Model Governance
- **Choice**: MLflow Tracking and Model Registry using a local directory store (`mlruns/`).
- **Rationale**: MLflow is open-source, vendor-agnostic, and standard across enterprise MLOps. It seamlessly logs parameters, metrics across training epochs/runs, registers versioned models, and standardizes model packaging (`mlflow.sklearn`).
- **Alternatives Considered**: Weights & Biases (requires cloud account/API keys); DVC (better for raw data versioning than interactive experiment dashboards).

### Decision 3: Pytest as Automated MLOps Quality Gates
- **Choice**: `pytest` for combined unit, data integrity, and model quality testing.
- **Rationale**: Allows testing both software logic (preprocessing transformations) and statistical properties (data shape, absence of NaNs in mandatory fields, inference latency, and minimum performance thresholds) within standard CI/CD pipelines.
- **Alternatives Considered**: Great Expectations standalone (heavier configuration overhead; pytest provides direct, flexible assertions and seamless CI runner integration).

### Decision 4: Evidently AI for Drift and Degradation Monitoring
- **Choice**: Evidently AI `Report` and `TestPreset` modules comparing reference training distributions with simulated production batches.
- **Rationale**: Provides statistical drift tests (Kolmogorov-Smirnov, Wasserstein distance) out of the box and outputs both machine-readable JSON metrics and executive-level interactive HTML reports.
- **Alternatives Considered**: Alibi Detect (complex API, less intuitive visual dashboard generation); custom statistical scripts (prone to maintenance overhead).

### Decision 5: Tabular Credit Risk Data Architecture
- **Choice**: Modular synthetic banking dataset generator modeled after standard Basel II / LendingClub credit risk datasets.
- **Rationale**: Enables deterministic generation with fixed random seeds, allowing reproducible simulation of macroeconomic shifts (e.g., rising interest rates, lower credit scores) to trigger and demonstrate Evidently drift alerts reliably.

## Risks / Trade-offs

- **[Risk] Library version friction between Evidently, MLflow, and Scikit-Learn** → *Mitigation*: Pin compatible semantic versions in `pyproject.toml` and lock dependencies via `uv lock`.
- **[Risk] Simulation of realistic drift in a local environment** → *Mitigation*: Create a dedicated drift-simulation generator that introduces calibrated distributional shifts in income and credit score for post-deployment verification.
- **[Risk] Execution friction during oral examination** → *Mitigation*: Provide automated one-command scripts (`uv run python ...` and `uv run pytest`) with clear log outputs and self-contained HTML reports.

## POE External Examiner Discussion Points
- **Architecture Overview**: Decoupling data preparation, model training/tracking, test gating, and continuous monitoring.
- **Tool Selection Rationale**: Why MLflow, pytest, and Evidently form a synergistic triumvirate for CI/CD and production feedback loops.
- **Data Drift vs. Concept Drift**: How Evidently detects input feature drift (covariate shift) before labels are observed versus target drift once ground-truth repayment outcomes arrive.
- **Continuous Integration / Continuous Deployment (CI/CD)**: How the pytest suite serves as an automated quality gate preventing sub-par models from promotion.
