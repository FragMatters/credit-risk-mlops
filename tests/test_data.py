"""Data integrity, schema conformity, and feature distribution test suite using pytest."""

from pathlib import Path
import pandas as pd
import pytest

DATA_PATH = Path("data/credit_risk.csv")

REQUIRED_NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "loan_amount",
    "interest_rate",
    "credit_score",
    "debt_to_income",
    "emp_length",
]

REQUIRED_CATEGORICAL_FEATURES = [
    "applicant_id",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

TARGET_COLUMN = "loan_status"


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """Fixture providing loaded credit risk dataframe."""
    assert DATA_PATH.exists(), f"Dataset file does not exist at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataset file is empty"
    return df


def test_schema_has_required_columns(dataset: pd.DataFrame) -> None:
    """Assert all expected feature and target columns exist in the dataset."""
    expected_columns = (
        REQUIRED_NUMERIC_FEATURES + REQUIRED_CATEGORICAL_FEATURES + [TARGET_COLUMN]
    )
    for col in expected_columns:
        assert col in dataset.columns, f"Mandatory column '{col}' missing from dataset schema"


def test_no_null_values_in_critical_columns(dataset: pd.DataFrame) -> None:
    """Assert zero missing values in primary key, financial inputs, and target."""
    critical_columns = [
        "applicant_id",
        "credit_score",
        "annual_income",
        "loan_amount",
        TARGET_COLUMN,
    ]
    for col in critical_columns:
        null_count = dataset[col].isnull().sum()
        assert null_count == 0, f"Critical column '{col}' contains {null_count} null values"


def test_credit_score_range(dataset: pd.DataFrame) -> None:
    """Assert credit scores conform to FICO bounds [350, 850]."""
    scores = dataset["credit_score"]
    assert (scores >= 350).all(), f"Credit score below minimum 350 found: {scores.min()}"
    assert (scores <= 850).all(), f"Credit score above maximum 850 found: {scores.max()}"


def test_positive_financial_values(dataset: pd.DataFrame) -> None:
    """Assert financial amounts and rates are strictly positive."""
    assert (dataset["annual_income"] > 0).all(), "Non-positive annual income detected"
    assert (dataset["loan_amount"] > 0).all(), "Non-positive loan amount detected"
    assert (dataset["interest_rate"] > 0).all(), "Non-positive interest rate detected"
    assert (dataset["debt_to_income"] > 0).all(), "Non-positive debt-to-income detected"


def test_age_and_employment_length(dataset: pd.DataFrame) -> None:
    """Assert applicant age and work experience are within realistic demographic bounds."""
    assert (dataset["age"] >= 18).all(), "Underage applicant detected (<18)"
    assert (dataset["age"] <= 100).all(), "Invalid applicant age detected (>100)"
    assert (dataset["emp_length"] >= 0).all(), "Negative employment length detected"
    assert (dataset["emp_length"] <= dataset["age"] - 14).all(), "Employment exceeds working lifetime"


def test_target_binary_classification(dataset: pd.DataFrame) -> None:
    """Assert target column contains only binary labels and balanced representation."""
    unique_labels = set(dataset[TARGET_COLUMN].unique())
    assert unique_labels == {0, 1}, f"Target labels must be {0, 1}, got {unique_labels}"

    default_rate = dataset[TARGET_COLUMN].mean()
    assert 0.05 <= default_rate <= 0.50, f"Default rate {default_rate:.2%} outside expected range [5%, 50%]"


def test_categorical_allowed_values(dataset: pd.DataFrame) -> None:
    """Assert categorical features contain only approved domain vocabulary."""
    allowed_grades = {"A", "B", "C", "D", "E"}
    allowed_defaults = {"Y", "N"}
    allowed_intents = {
        "DEBTCONSOLIDATION",
        "PERSONAL",
        "HOMEIMPROVEMENT",
        "EDUCATION",
        "VENTURE",
        "MEDICAL",
    }

    assert set(dataset["loan_grade"].unique()).issubset(allowed_grades), "Unapproved loan grade detected"
    assert set(dataset["cb_person_default_on_file"].unique()).issubset(allowed_defaults), "Invalid default flag"
    assert set(dataset["loan_intent"].unique()).issubset(allowed_intents), "Unapproved loan intent detected"
