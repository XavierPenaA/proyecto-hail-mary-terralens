"""HTTP contract of the API: shapes, validation and error codes (no network access needed)."""

import pytest
from fastapi.testclient import TestClient

from api import app

VERDICT_CODES = {"sig_increase", "sig_decrease", "marginal_increase", "marginal_decrease", "not_significant", "autocorr_rejected"}


@pytest.fixture(scope="module")
def client():
    # Not used as a context manager: the startup warm-up thread is not needed in tests.
    return TestClient(app)


def test_index_and_static_assets_are_served(client):
    assert client.get("/").status_code == 200
    for path in ("/static/js/main.js", "/static/css/app.css", "/static/i18n/es.json", "/static/vendor/chart.umd.js"):
        assert client.get(path).status_code == 200, path
    for name in ("instrument-sans", "instrument-sans-extended", "ibm-plex-mono-regular", "ibm-plex-mono-semibold"):
        font = client.get(f"/static/fonts/{name}.woff2")
        assert font.status_code == 200, name
        assert font.headers["content-type"] == "font/woff2", name
        assert font.content.startswith(b"wOF2"), name


def test_catalog_lists_regions_variables_and_computed_cases(client):
    body = client.get("/api/catalog").json()
    assert len(body["regions"]) == 11
    assert len(body["variables"]) == 6
    assert body["map_variables"] == ["T2M", "PRECTOTCORR", "GWETROOT"]
    for case in body["cases"]:
        assert case["finding"]["en"] and case["finding"]["es"]
        assert case["steps"] and all(step["view"].startswith("tab-") for step in case["steps"])


def test_full_dashboard_contract(client):
    res = client.get("/api/full-dashboard/EC_Andes_Paute", params={"variable": "GWETROOT"})
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"]["code"] in VERDICT_CODES
    assert body["homogeneity"]["preferred_model"] in {"step", "trend", "inconclusive"}
    assert len(body["monthly_series"]) == body["n_months"]
    assert [r["variable"] for r in body["auto_scan"]["rows"]] == list(client.get("/api/catalog").json()["variables"])
    assert body["ml"]["lagged_driver_model"]["available"]


def test_solar_window_is_moved_to_the_ceres_era_when_the_guard_is_on(client):
    body = client.get("/api/full-dashboard/EC_Coast_Guayas", params={"variable": "ALLSKY_SFC_SW_DWN"}).json()
    assert body["year_start"] == 2001
    assert body["window"]["start_moved_for_variable"]
    off = client.get(
        "/api/full-dashboard/EC_Coast_Guayas", params={"variable": "ALLSKY_SFC_SW_DWN", "homogeneous_ceres": False}
    ).json()
    assert off["year_start"] == 1984
    assert off["known_break"]["break_date"] == "2001-01"
    assert "known_splice_in_window" in off["warnings"]

@pytest.mark.parametrize("path", ["/api/full-dashboard/EC_Andes_Paute", "/api/dipole"])
@pytest.mark.parametrize("start,end", [(1981, 1990), (1993, 2002)])
def test_solar_window_must_be_long_enough_after_applying_the_guard(client, path, start, end):
    result = client.get(path, params={"variable": "ALLSKY_SFC_SW_DWN", "start_year": start, "end_year": end})
    assert result.status_code == 422
    detail = result.json()["detail"]
    assert detail["code"] == "short_effective_window"
    assert detail["first_year"] == 2001


def test_dashboard_counts_only_months_removed_from_the_solar_series(client):
    path = "/api/full-dashboard/EC_Coast_Guayas"
    params = {"variable": "ALLSKY_SFC_SW_DWN"}
    baseline = client.get(path, params=params).json()
    filtered = client.get(path, params={**params, "exclude_ml_extremes": True}).json()
    before = {row["date"] for row in baseline["monthly_series"]}
    after = {row["date"] for row in filtered["monthly_series"]}
    assert 0 < len(before - after) == filtered["filters"]["n_excluded_months"]


def test_dipole_uses_the_requested_eras_when_solar_guard_is_off(client):
    res = client.get("/api/dipole", params={
        "variable": "ALLSKY_SFC_SW_DWN", "start_year": 1984, "end_year": 1993, "homogeneous_ceres": False,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["stats_a"]["period"] == body["stats_b"]["period"] == [1984, 1993]
    assert body["series"][0]["date"] == "1984-01"
    assert body["series"][-1]["date"] == "1993-12"


@pytest.mark.parametrize(
    "path, params, status",
    [
        ("/api/full-dashboard/NOPE", {}, 404),
        ("/api/full-dashboard/EC_Andes_Paute", {"variable": "FOO"}, 422),
        ("/api/full-dashboard/EC_Andes_Paute", {"start_year": 2015, "end_year": 1995}, 422),
        ("/api/full-dashboard/EC_Andes_Paute", {"start_year": 2020, "end_year": 2025}, 422),
        ("/api/full-dashboard/EC_Andes_Paute", {"start_year": 1970}, 422),
        ("/api/dipole", {"region_a": "XX"}, 422),
        ("/api/dipole", {"region_a": "EC_Coast_Guayas", "region_b": "EC_Coast_Guayas"}, 422),
        ("/api/multiscale", {"grid_variable": "RH2M"}, 422),
        ("/api/live-probe", {"lat": 0, "lon": 0, "variable": "FOO"}, 422),
        ("/api/live-probe", {"lat": 95, "lon": 0}, 422),
        ("/api/cube-ml", {}, 404),
    ],
)
def test_invalid_requests_are_rejected(client, path, params, status):
    assert client.get(path, params=params).status_code == status


def test_dipole_reports_relations_with_effective_sample_size(client):
    body = client.get("/api/dipole").json()
    assert body["enso_relation"] in {"opposite", "same_sign", "not_both_significant"}
    assert body["trend_relation"] in {"opposite", "same_direction", "not_both_significant"}
    assert body["cross_correlation"]["n_eff"] <= body["cross_correlation"]["n"]


def test_multiscale_flags_the_modis_fallback(client):
    body = client.get("/api/multiscale", params={"region_id": "EC_Galapagos"}).json()
    assert body["modis"]["is_fallback"]
    assert len(body["grid_cells"]) == 96


def test_responses_are_compressed(client):
    res = client.get("/api/full-dashboard/EC_Andes_Paute", headers={"Accept-Encoding": "gzip"})
    assert res.headers.get("content-encoding") == "gzip"


def test_manifest_includes_the_persistence_baseline(client):
    manifest = client.get("/api/ml-manifest").json()
    names = [m["model_name"] for m in manifest["nowcast_benchmark"]["leaderboard"]]
    assert names[0] == "PersistenceBaseline"
    assert {b["horizon_months"] for b in manifest["forecast_benchmarks"]} == {1, 3}
