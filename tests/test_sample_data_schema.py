
import os
import pandas as pd
import pytest

SAMPLE_FILE = "sample_data.csv"

def test_sample_schema():
    if not os.path.exists(SAMPLE_FILE):
        pytest.skip("sample_data.csv missing; provide a sample or data contract.")
    df = pd.read_csv(SAMPLE_FILE)
    assert "month" in df.columns, "Expected 'month' column"
    assert "kwh" in df.columns, "Expected 'kwh' column"
