"""
The guided-case narratives rest on data-dependent claims. These tests pin each claim to the engine,
so a data refresh that breaks a narrative fails here instead of silently shipping a wrong story.
"""

import re

import pytest

from core.case_studies import CASES, case_finding, public_cases


@pytest.mark.parametrize("case_id", [c["id"] for c in CASES])
def test_every_case_produces_a_bilingual_finding(case_id):
    text = case_finding(case_id)["text"]
    assert len(text["en"]) > 80 and len(text["es"]) > 80
    # Both languages quote the same numbers.
    numbers = lambda s: sorted(re.findall(r"[−+]?\d+\.\d+", s))  # noqa: E731
    assert numbers(text["en"]) == numbers(text["es"])


def test_enso_dipole_claims():
    m = case_finding("enso_dipole")["metrics"]
    assert m["coast_enso"]["r"] > 0 and m["coast_enso"]["q"] < 0.05
    assert m["amazon_enso"]["r"] < 0 and m["amazon_enso"]["q"] < 0.05
    assert m["coast_trend"]["verdict"] == "not_significant"
    assert m["coast_peak"]["date"].startswith("1997")


def test_autocorrelation_trap_claims():
    m = case_finding("autocorrelation_trap")["metrics"]
    assert m["p_naive"] < 0.05, "the naive regression must look significant for the trap to exist"
    assert m["lag1_r"] > 0.85
    assert m["verdict"] in {"autocorr_rejected", "not_significant"}


def test_sensor_splice_claims():
    m = case_finding("sensor_splice")["metrics"]
    for site in ("coast", "pantanal"):
        assert m[site]["known_break"]["break_date"] == "2001-01"
        assert m[site]["known_break"]["is_significant"]
        assert m[site]["ceres_period"][0] == 2001
    assert m["coast"]["full"]["slope_decade"] < 0 < m["pantanal"]["full"]["slope_decade"]
    assert m["coast"]["ceres"]["verdict"] == "not_significant"


def test_pantanal_claims():
    m = case_finding("pantanal_compound")["metrics"]
    assert all(t["verdict"].startswith("sig_") for t in m["trends"].values())


def test_arctic_claims():
    m = case_finding("arctic_vs_delta")["metrics"]
    assert m["ratio"] > 1.5
    assert m["arctic"]["verdict"] == "sig_increase"


def test_napo_claims():
    m = case_finding("napo_step")["metrics"]
    assert m["coincident_breaks"] is not None
    assert len(m["coincident_breaks"]["variables"]) >= 3
    assert m["coincident_breaks"]["date"].startswith("2007")


def test_public_cases_are_numbered_in_order():
    assert [c["number"] for c in public_cases()] == list(range(1, len(CASES) + 1))
