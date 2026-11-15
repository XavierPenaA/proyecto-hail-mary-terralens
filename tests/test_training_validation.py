import pandas as pd
import pytest

from train_models import build_feature_folds, standardize_on_training_era


def test_temporal_folds_keep_entire_months_and_respect_forecast_lead():
    dates = pd.period_range("1990-01", periods=36, freq="M").astype(str)
    rows = pd.DataFrame({"date": dates.repeat(3)})
    cutoffs = []

    def build(cutoff):
        cutoffs.append(cutoff)
        return rows

    for (training, validation), cutoff in zip(build_feature_folds(rows["date"], build, gap=2), cutoffs):
        assert training["date"].max() == cutoff
        assert (pd.Period(validation["date"].min(), "M") - pd.Period(cutoff, "M")).n == 3
        assert set(validation.groupby("date").size()) == {3}
    assert len(cutoffs) == 5


def test_validation_values_do_not_fit_the_training_climatology():
    observations = pd.DataFrame({
        "date": ["1990-01", "1991-01", "1992-01"],
        "year": [1990, 1991, 1992],
        "month": [1, 1, 1],
        "T2M": [10, 20, 200],
    })
    result = standardize_on_training_era(observations, ["T2M"], "1991-12")
    assert result["T2M_z"][:2].mean() == pytest.approx(0)
    assert result["T2M_z"].iloc[-1] > 20
