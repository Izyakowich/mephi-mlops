import pytest
import pandas as pd
import numpy as np
from src.features.build_features import build_features
from src.models.pipeline import get_pipeline

def test_build_features():
    df = pd.DataFrame({
        'AGE': [25, 45, 60],
        'PAY_AMT1': [100, 200, 300],
        'PAY_AMT2': [100, 200, 300],
        'PAY_AMT3': [100, 200, 300],
        'PAY_AMT4': [100, 200, 300],
        'PAY_AMT5': [100, 200, 300],
        'PAY_AMT6': [100, 200, 300],
        'BILL_AMT1': [1000, 2000, 3000]
    })
    result = build_features(df)
    assert 'AGE_BIN' in result.columns
    assert 'PAY_AMT_MEAN' in result.columns
    assert result['PAY_AMT_MEAN'].iloc[0] == 100

def test_pipeline_creation():
    pipe = get_pipeline("gb")
    assert pipe is not None