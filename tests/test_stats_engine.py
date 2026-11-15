"""Unit tests of the statistical engine on synthetic series with known answers."""

import numpy as np
import pandas as pd
import pytest

from core.stats_engine import (
    analyze_single_variable,
    benjamini_hochberg_fdr,
    compare_step_vs_trend,
    compute_climatology_and_seasonal_trends,
    correlation_with_neff,
    deseasonalize_acker,
    known_breakpoint_test,
    mann_kendall_hamed_rao,
    ols_with_santer_correction,
    pettitt_change_point_test,
)

N_YEARS = 40


def monthly_frame(values: np.ndarray, start_year: int = 1981) -> pd.DataFrame:
    n = len(values)
    years = start_year + np.arange(n) // 12
    months = np.arange(n) % 12 + 1
    return pd.DataFrame(
        {
            "date": [f"{y:04d}-{m:02d}" for y, m in zip(years, months)],
            "year": years,
            "month": months,
            "decimal_year": years + (months - 0.5) / 12.0,
            "X": values,
        }
    )


def ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


def test_deseasonalize_removes_the_seasonal_cycle():
    months = np.tile(np.arange(12), N_YEARS)
    df = monthly_frame(10 * np.sin(2 * np.pi * months / 12) + 3.0)
    des = deseasonalize_acker(df, "X")
    assert np.allclose(des["anomaly"], 0.0)


def test_mann_kendall_detects_a_clear_trend_and_its_slope():
    rng = np.random.default_rng(1)
    t = np.arange(200, dtype=float)
    res = mann_kendall_hamed_rao(t, 0.05 * t + rng.normal(0, 0.5, 200))
    assert res["p_hamed_rao"] < 1e-6
    assert res["theil_sen_yr"] == pytest.approx(0.05, abs=0.01)


def test_hamed_rao_is_more_conservative_than_naive_mk_for_persistent_noise():
    rng = np.random.default_rng(7)
    x = ar1(480, 0.9, rng)
    res = mann_kendall_hamed_rao(np.arange(480, dtype=float), x)
    assert res["variance_inflation_factor"] > 1.5
    assert res["p_hamed_rao"] > res["p_original"]


def test_santer_reduces_the_effective_sample_size():
    rng = np.random.default_rng(3)
    t = np.arange(480, dtype=float)
    res = ols_with_santer_correction(t, ar1(480, 0.8, rng))
    assert res["lag1_r"] > 0.6
    assert res["n_eff"] < 0.25 * res["n"]
    assert res["p_santer"] >= res["p_naive"]


def test_pettitt_finds_a_step_and_bootstrap_controls_persistence():
    rng = np.random.default_rng(11)
    step = np.r_[np.zeros(240), np.full(240, 1.5)] + rng.normal(0, 0.5, 480)
    dates = monthly_frame(step)["date"].tolist()
    res = pettitt_change_point_test(dates, step, n_boot=199)
    assert abs(res["change_index"] - 239) <= 3
    assert res["p_value"] <= 0.01

    noise = ar1(480, 0.95, rng)
    res_noise = pettitt_change_point_test(dates, noise, n_boot=199)
    # The asymptotic formula assumes independence and is far too confident for persistent noise.
    assert res_noise["p_value"] > res_noise["p_value_asymptotic"]


def test_known_breakpoint_measures_the_documented_step():
    rng = np.random.default_rng(5)
    values = np.r_[np.zeros(200), np.full(200, -0.7)] + rng.normal(0, 0.2, 400)
    dates = monthly_frame(values, start_year=1984)["date"].tolist()
    res = known_breakpoint_test(dates, values, dates[200], n_boot=199)
    assert res["step_delta"] == pytest.approx(-0.7, abs=0.08)
    assert res["is_significant"]
    assert known_breakpoint_test(dates, values, "2100-01") is None


def test_step_versus_trend_prefers_the_true_model():
    rng = np.random.default_rng(2)
    t = np.arange(480, dtype=float)
    step = np.r_[np.zeros(240), np.ones(240)] + rng.normal(0, 0.3, 480)
    trend = t / 240.0 + rng.normal(0, 0.3, 480)
    assert compare_step_vs_trend(t, step, 239)["preferred_model"] == "step"
    assert compare_step_vs_trend(t, trend, 239)["preferred_model"] == "trend"


def test_correlation_with_neff_penalizes_two_persistent_series():
    rng = np.random.default_rng(9)
    common = ar1(500, 0.95, rng)
    a = common + rng.normal(0, 1.0, 500)
    b = common + rng.normal(0, 1.0, 500)
    res = correlation_with_neff(a, b)
    assert res["n_eff"] < 0.5 * res["n"]


def test_djf_uses_december_of_the_previous_year():
    values = np.zeros(12 * 10)
    df = monthly_frame(values)
    df.loc[(df["year"] == 1985) & (df["month"] == 12), "X"] = 120.0  # belongs to DJF 1986
    des = deseasonalize_acker(df, "X")
    rows = des.assign(season_year=des["year"] + (des["month"] == 12).astype(int))
    djf = rows[rows["month"].isin([12, 1, 2])].groupby("season_year")["anomaly"].mean()
    assert djf.idxmax() == 1986
    seasons = compute_climatology_and_seasonal_trends(des)["seasonal_trends"]
    djf_row = next(s for s in seasons if s["season"] == "DJF")
    assert djf_row["n_years"] == 9  # incomplete seasons at both ends are dropped


def test_benjamini_hochberg_matches_the_textbook_example():
    q = benjamini_hochberg_fdr([0.01, 0.04, 0.03, 0.005])
    assert q == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert benjamini_hochberg_fdr([]) == []


def test_verdict_rejects_a_trend_that_only_memory_creates():
    rng = np.random.default_rng(21)
    df = monthly_frame(ar1(12 * N_YEARS, 0.97, rng) * 0.05)
    res = analyze_single_variable(df, "X")
    assert not res["verdict"]["is_significant"]
