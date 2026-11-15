"""Model quality gates, performance threshold, and inference latency test suite using pytest."""

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

MODEL_PATH = Path("models/best_model.joblib")
TEST_DATA_PATH = Path("data/test.csv")


@pytest.fixture(scope="module")
def model_pipeline():
    """Load serialized champion model pipeline."""
    assert MODEL_PATH.exists(), f"Trained model not found at {MODEL_PATH}. Run training first."
    pipeline = joblib.load(MODEL_PATH)
    assert pipeline is not None, "Loaded model is None"
    return pipeline


@pytest.fixture(scope="module")
def test_dataset():
    """Load evaluation test split."""
    assert TEST_DATA_PATH.exists(), f"Test split not found at {TEST_DATA_PATH}"
    df = pd.read_csv(TEST_DATA_PATH)
    assert not df.empty, "Test split is empty"
    return df


def test_model_pipeline_inference(model_pipeline, test_dataset) -> None:
    """Assert pipeline executes inference and outputs valid class and probability."""
    sample = test_dataset.drop(columns=["loan_status"]).iloc[:5]
    preds = model_pipeline.predict(sample)
    probs = model_pipeline.predict_proba(sample)[:, 1]

    assert len(preds) == 5, "Incorrect prediction array length"
    assert set(preds).issubset({0, 1}), f"Predictions must be binary, got {preds}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]"


def test_model_performance_gate_roc_auc(model_pipeline, test_dataset) -> None:
    """Assert model passes the POE minimum performance quality gate (ROC-AUC >= 0.75)."""
    X_test = test_dataset.drop(columns=["loan_status"])
    y_test = test_dataset["loan_status"]

    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    assert (
        roc_auc >= 0.75
    ), f"Model failed POE performance quality gate: ROC-AUC {roc_auc:.4f} < 0.75"


def test_inference_latency_gate(model_pipeline, test_dataset) -> None:
    """Assert single-record inference latency is strictly under 50ms (SLO gate)."""
    sample = test_dataset.drop(columns=["loan_status"]).iloc[:1]

    # Warmup
    _ = model_pipeline.predict_proba(sample)

    # Benchmark 50 iterations
    latencies_ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model_pipeline.predict_proba(sample)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    p95_latency = np.percentile(latencies_ms, 95)
    assert (
        p95_latency < 50.0
    ), f"P95 inference latency {p95_latency:.2f}ms exceeds SLO threshold of 50ms"


def test_monotonic_risk_property(model_pipeline) -> None:
    """Domain property test: A low-risk borrower must have lower default prob than high-risk borrower."""
    low_risk = pd.DataFrame(
        [
            {
                "age": 45,
                "annual_income": 150000.0,
                "loan_amount": 10000.0,
                "loan_intent": "HOMEIMPROVEMENT",
                "loan_grade": "A",
                "interest_rate": 6.5,
                "credit_score": 810,
                "debt_to_income": 0.10,
                "cb_person_default_on_file": "N",
                "emp_length": 15,
            }
        ]
    )

    high_risk = pd.DataFrame(
        [
            {
                "age": 23,
                "annual_income": 25000.0,
                "loan_amount": 25000.0,
                "loan_intent": "DEBTCONSOLIDATION",
                "loan_grade": "E",
                "interest_rate": 22.0,
                "credit_score": 480,
                "debt_to_income": 1.20,
                "cb_person_default_on_file": "Y",
                "emp_length": 1,
            }
        ]
    )

    low_risk_prob = float(model_pipeline.predict_proba(low_risk)[0, 1])
    high_risk_prob = float(model_pipeline.predict_proba(high_risk)[0, 1])

    assert (
        low_risk_prob < high_risk_prob
    ), f"Domain violation: low-risk prob ({low_risk_prob:.3f}) >= high-risk prob ({high_risk_prob:.3f})"
