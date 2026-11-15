"""Credit Risk ML Training Pipeline with MLflow experiment tracking and model registry.

Trains baseline and optimized classification models for loan default prediction,
evaluates performance metrics, logs artifacts to MLflow, and registers candidate models.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Disable agent hint and configure writable matplotlib config
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "loan_amount",
    "interest_rate",
    "credit_score",
    "debt_to_income",
    "emp_length",
]

CATEGORICAL_FEATURES = [
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

TARGET_COLUMN = "loan_status"


def build_preprocessor() -> ColumnTransformer:
    """Construct sklearn ColumnTransformer for numeric and categorical features."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def train_and_evaluate(
    data_path: str = "data/credit_risk.csv",
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "credit-risk-scoring",
    min_roc_auc_gate: float = 0.75,
) -> dict:
    """Execute training pipeline, log runs to MLflow, and register passing model."""
    df = pd.read_csv(data_path)

    # Stratified Train/Test split
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save splits for reference in monitoring and testing
    data_dir = Path(data_path).parent
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    # Configure MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42
        ),
    }

    best_model_name = None
    best_roc_auc = 0.0
    best_pipeline = None
    training_results = {}

    artifacts_dir = Path("reports/mlflow_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for model_name, classifier in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            preprocessor = build_preprocessor()
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", classifier)]
            )

            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            # Compute metrics
            roc_auc = float(roc_auc_score(y_test, y_pred_proba))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            precision = float(precision_score(y_test, y_pred, zero_division=0))
            recall = float(recall_score(y_test, y_pred, zero_division=0))
            loss = float(log_loss(y_test, y_pred_proba))

            metrics = {
                "roc_auc": round(roc_auc, 4),
                "f1_score": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "log_loss": round(loss, 4),
            }

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            for k, v in classifier.get_params().items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(f"model__{k}", v)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Confusion matrix plot artifact
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Default", "Default"])
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            plt.title(f"Confusion Matrix: {model_name}")
            cm_plot_path = artifacts_dir / f"{model_name}_confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(cm_plot_path, dpi=120)
            plt.close(fig)
            mlflow.log_artifact(str(cm_plot_path), artifact_path="plots")

            # Classification report artifact
            clf_rep = classification_report(y_test, y_pred, output_dict=True)
            rep_path = artifacts_dir / f"{model_name}_classification_report.json"
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(clf_rep, f, indent=2)
            mlflow.log_artifact(str(rep_path), artifact_path="reports")

            # Log model
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            # Evaluate promotion gate
            passed_gate = roc_auc >= min_roc_auc_gate
            mlflow.set_tag("gate_passed", str(passed_gate))
            mlflow.set_tag("domain", "FinTech-CreditRisk")

            training_results[model_name] = {
                "run_id": run.info.run_id,
                "metrics": metrics,
                "passed_gate": passed_gate,
            }

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_name = model_name
                best_pipeline = pipeline

    # Register best model in MLflow Model Registry if threshold met
    registry_name = "CreditRiskClassifier"
    if best_pipeline is not None and best_roc_auc >= min_roc_auc_gate:
        best_run_id = training_results[best_model_name]["run_id"]
        model_uri = f"runs:/{best_run_id}/model"
        reg_model = mlflow.register_model(
            model_uri=model_uri,
            name=registry_name,
        )
        training_results["registered_model"] = {
            "name": registry_name,
            "version": reg_model.version,
            "source_run_id": best_run_id,
            "best_model": best_model_name,
            "best_roc_auc": round(best_roc_auc, 4),
        }

        # Save local artifact for testing suite
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, models_dir / "best_model.joblib")
        print(
            f"Successfully registered model '{registry_name}' (Version {reg_model.version}) with ROC-AUC {best_roc_auc:.4f}"
        )

    return training_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk models with MLflow tracking.")
    parser.add_argument("--data", type=str, default="data/credit_risk.csv", help="Input dataset")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--gate", type=float, default=0.75, help="Minimum ROC-AUC deployment gate")
    args = parser.parse_args()

    results = train_and_evaluate(
        data_path=args.data,
        tracking_uri=args.tracking_uri,
        min_roc_auc_gate=args.gate,
    )

    print("\n" + "=" * 60)
    print("MLFLOW EXPERIMENT TRAINING SUMMARY")
    print("=" * 60)
    for model_name, info in results.items():
        if model_name != "registered_model":
            print(f"Model: {model_name}")
            print(f"  Run ID: {info['run_id']}")
            print(f"  ROC-AUC: {info['metrics']['roc_auc']}")
            print(f"  F1-Score: {info['metrics']['f1_score']}")
            print(f"  Gate Passed: {info['passed_gate']}")
    if "registered_model" in results:
        reg = results["registered_model"]
        print(f"\nModel Registry:")
        print(f"  Name: {reg['name']} (v{reg['version']})")
        print(f"  Best Model: {reg['best_model']} (ROC-AUC: {reg['best_roc_auc']})")


if __name__ == "__main__":
    main()
