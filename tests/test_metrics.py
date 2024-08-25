
import pandas as pd
import numpy as np
import pytest

from src.metrics import compute_eui, baseline_delta

def test_compute_eui_ok():
    df = pd.DataFrame({"kwh":[100,200,300], "area_m2":[1000,1000,1000]})
    assert abs(compute_eui(df) - (600/1000)) < 1e-9

def test_compute_eui_invalid_area():
    df = pd.DataFrame({"kwh":[100,200], "area_m2":[0,0]})
    with pytest.raises(ValueError):
        compute_eui(df)

def test_baseline_delta_with_baseline():
    df = pd.DataFrame({"kwh":[100,200], "baseline":[120,220]})
    assert baseline_delta(df) == (340 - 300)

def test_baseline_delta_no_baseline():
    df = pd.DataFrame({"kwh":[100,200]})
    assert baseline_delta(df) == 0.0
