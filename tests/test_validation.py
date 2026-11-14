import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.data.validation import validate_data


@pytest.fixture
def sample_data():
    return pd.read_csv("tests/fixtures/credit_sample.csv").copy()


def test_valid_data(sample_data):
    validated = validate_data(sample_data)
    assert len(validated) == 100


def test_invalid_age(sample_data):
    sample_data.loc[0, "AGE"] = 150

    with pytest.raises(SchemaErrors):
        validate_data(sample_data)


def test_missing_values(sample_data):
    sample_data.loc[0, "LIMIT_BAL"] = None

    with pytest.raises(SchemaErrors):
        validate_data(sample_data)


def test_invalid_target(sample_data):
    sample_data.loc[0, "default.payment.next.month"] = 5

    with pytest.raises(SchemaErrors):
        validate_data(sample_data)


def test_missing_column(sample_data):
    sample_data = sample_data.drop(columns=["AGE"])

    with pytest.raises(SchemaErrors):
        validate_data(sample_data)
