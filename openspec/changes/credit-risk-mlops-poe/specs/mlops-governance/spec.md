# Spec Delta

## Purpose
Provides experiment tracking and model registry through MLflow, automated validation testing via pytest, and continuous drift monitoring through Evidently AI for the credit risk machine learning lifecycle.

## ADDED Requirements

### Requirement: MLflow Experiment Tracking and Model Registry
The system SHALL log model hyperparameters, evaluation metrics (ROC-AUC, F1-score, Precision, Recall), confusion matrices, and serialized model artifacts to an MLflow experiment run and register production-ready candidate models in the MLflow Model Registry.

#### Scenario: Model run logging and registration
- **WHEN** a model training run executes
- **THEN** parameters, performance metrics, and the model artifact SHALL be recorded in MLflow, and models meeting production quality gates SHALL be registered.

### Requirement: Automated Pytest Quality and Gate Suite
The system SHALL execute a comprehensive pytest test suite validating input data schemas, null tolerance, model inference latency, and minimum classification performance gates.

#### Scenario: Test suite execution on candidate model
- **WHEN** pytest is executed against the candidate model and dataset
- **THEN** all data validation and model performance assertions SHALL pass with zero test failures before promotion.

### Requirement: Evidently AI Data and Model Drift Monitoring
The system SHALL compare production inference batches against the reference baseline training dataset using Evidently AI to detect feature distribution shifts and target drift.

#### Scenario: Drift report generation for production batch
- **WHEN** new inference batches are evaluated by the monitoring module
- **THEN** an Evidently drift report SHALL be generated containing drift detection metrics and an interactive HTML report visualization.
