"""The cached datasets the observatory ships with are complete and consistent with the catalog."""

import pytest

from core.catalog import DATA_END_YEAR, DATA_START_YEAR, REGIONS, VARIABLES
from core.connectors import (
    load_ecuador_regional_grid_trends,
    load_meteomatics_1km_cache,
    load_modis_ndvi,
    load_noaa_oni,
    load_planetary_computer_stac,
    load_region_nasa_power,
)


@pytest.mark.parametrize("region_id", list(REGIONS))
def test_every_region_has_the_full_monthly_record(region_id):
    df, meta = load_region_nasa_power(region_id)
    assert df["date"].iloc[0] == f"{DATA_START_YEAR}-01"
    assert df["date"].iloc[-1] == f"{DATA_END_YEAR}-12"
    for code, variable in VARIABLES.items():
        assert code in df.columns
        first_valid = df.loc[df[code].notna(), "year"].min()
        assert first_valid == variable["first_year"], f"{region_id} {code} starts in {first_valid}"
    assert meta["source_mode"] == "cache"


def test_oni_covers_the_data_period():
    oni = load_noaa_oni()
    assert oni["year"].min() <= DATA_START_YEAR
    assert oni["year"].max() >= DATA_END_YEAR
    assert set(oni["month"]) == set(range(1, 13))


@pytest.mark.parametrize("variable", [v for v, meta in VARIABLES.items() if meta["map"]])
def test_grid_trends_cover_96_cells_with_fdr(variable):
    grid = load_ecuador_regional_grid_trends(variable)
    assert len(grid) == 96
    assert (grid["q_fdr"] >= grid["p_hamed_rao"] - 1e-9).all()
    assert grid["significant"].equals(grid["q_fdr"] < 0.05)


def test_satellite_caches_match_the_catalog_flags():
    meteomatics = load_meteomatics_1km_cache()
    for region_id, region in REGIONS.items():
        assert (not load_modis_ndvi(region_id).empty) == region["has_modis_250m"], region_id
        assert (region_id in meteomatics) == region["has_meteomatics_1km"], region_id
        assert load_planetary_computer_stac(region_id), region_id
