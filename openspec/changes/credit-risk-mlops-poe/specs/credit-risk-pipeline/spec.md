# Spec Delta

## Purpose
Provides an end-to-end data ingestion, exploratory data analysis, feature preprocessing, and machine learning modeling pipeline for loan default prediction in the FinTech banking domain.

## ADDED Requirements

### Requirement: Credit Risk Dataset Ingestion and Validation
The system SHALL ingest credit applicant records containing demographic, financial, and credit history features while validating schema consistency and target label definitions.

#### Scenario: Ingestion of valid credit records
- **WHEN** raw credit applicant data is ingested through the pipeline
- **THEN** the pipeline SHALL validate column datatypes, reject malformed records, and output standardized data splits for training and evaluation.

### Requirement: Automated Exploratory Data Analysis Generation
The system SHALL compute exploratory statistical summaries, feature missingness rates, class imbalance ratios, and correlation matrices for the ingested credit risk dataset.

#### Scenario: Execution of automated EDA
- **WHEN** the EDA module is executed against the baseline credit dataset
- **THEN** it SHALL generate statistical summaries of loan amounts, debt-to-income ratios, and default rates, and export an EDA summary report.

### Requirement: Credit Risk Model Training and Inference
The system SHALL preprocess tabular credit features, fit classification models to predict the probability of loan default, and produce binary default predictions based on a configurable decision threshold.

#### Scenario: Model training and probability prediction
- **WHEN** training data is passed to the credit risk model trainer
- **THEN** the model SHALL output calibrated default probabilities and class predictions with test ROC-AUC exceeding 0.75.
