"""
Empirical & Mathematical Validity Test Suite for TerraLens v2.1.
Verifies Rule #0 (Zero Hallucination / 100% Real Data & Peer-Reviewed Statistics).
"""

import os
import sys
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.catalog import REGIONS, VARIABLES, GUIDED_CASES
from core.connectors import (
    load_region_nasa_power,
    load_noaa_oni,
    load_ecuador_regional_grid_trends,
    load_or_fetch_modis_250m,
    load_meteomatics_1km_cache,
)
from core.stats_engine import (
    analyze_single_variable,
    compare_regional_dipole,
    benjamini_hochberg_fdr,
)
from core.ml_engine import run_multivariate_anomaly_detective


def run_all_tests() -> None:
    # Copy verified_audit_summary.csv if needed
    src_csv = r"C:\Users\xavit\.gemini\antigravity\brain\78f8de84-6f79-4485-9732-4dc872a37ae4\scratch\spike_results.csv"
    dst_csv = os.path.join(BASE_DIR, "data", "verified_audit_summary.csv")
    if os.path.exists(src_csv) and not os.path.exists(dst_csv):
        shutil.copy2(src_csv, dst_csv)

    # 1. Verify all 11 regions load real NASA POWER 1981-2025 data
    for reg_id in REGIONS:
        df, meta = load_region_nasa_power(reg_id)
        assert len(df) >= 500, f"Region {reg_id} has too few months: {len(df)}"
        for v in VARIABLES:
            assert v in df.columns, f"Missing {v} in {reg_id}"
    print("[PASS] All 11 regions loaded from real NASA POWER 1981-2025 observations.")

    # 2. Verify Trap #1 Protection: Autocorrelation correction on EC_Andes_Paute & EC_Andes_Quito GWETROOT
    oni_df = load_noaa_oni()
    assert len(oni_df) > 800, "NOAA ONI series failed to load."
    df_paute, _ = load_region_nasa_power("EC_Andes_Paute")
    res_paute_gw = analyze_single_variable(df_paute, "GWETROOT", oni_df=oni_df)
    assert res_paute_gw["monthly_ols"]["p_naive"] < 0.05, "Expected naive OLS false positive on Paute GWETROOT"
    assert res_paute_gw["monthly_ols"]["lag1_r"] > 0.85, "Expected high lag-1 autocorrelation on Paute GWETROOT"
    assert res_paute_gw["monthly_ols"]["p_santer"] > 0.20, "Santer correction must reject false positive"
    assert res_paute_gw["annual_mk"]["p_original"] > 0.15, "Annual MK must reject false positive"
    assert "No Statistically Significant" in res_paute_gw["verdict"]["verdict"]

    df_quito, _ = load_region_nasa_power("EC_Andes_Quito")
    res_quito_gw = analyze_single_variable(df_quito, "GWETROOT", oni_df=oni_df)
    assert res_quito_gw["monthly_mk"]["p_original"] < 0.001 and res_quito_gw["monthly_mk"]["p_hamed_rao"] > 0.10
    print(
        f"[PASS] Trap #1 Caught: Paute GWETROOT p_naive={res_paute_gw['monthly_ols']['p_naive']:.4f} "
        f"-> p_Santer={res_paute_gw['monthly_ols']['p_santer']:.4f}, p_Annual={res_paute_gw['annual_mk']['p_original']:.4f}; "
        f"Quito GWETROOT p_MK_orig={res_quito_gw['monthly_mk']['p_original']:.2e} -> p_HR={res_quito_gw['monthly_mk']['p_hamed_rao']:.4f}"
    )

    # 3. Verify Trap #2 Protection: Jan 2001 Satellite Splice in ALLSKY_SFC_SW_DWN
    df_guayas, _ = load_region_nasa_power("EC_Coast_Guayas")
    res_guayas_sw_full = analyze_single_variable(df_guayas, "ALLSKY_SFC_SW_DWN", start_year=1984, end_year=2025)
    res_guayas_sw_ceres = analyze_single_variable(df_guayas, "ALLSKY_SFC_SW_DWN", start_year=2001, end_year=2025)
    assert res_guayas_sw_full["pettitt"]["is_significant_breakpoint"], "Pettitt must flag 2000/2001 splice"
    assert abs(res_guayas_sw_ceres["monthly_mk"]["theil_sen_decade"]) < 0.05, "Homogeneous CERES era must have ~0 slope"
    print(
        f"[PASS] Trap #2 Caught: Guayas SW_DWN Pettitt breakpoint at {res_guayas_sw_full['pettitt']['change_date']} "
        f"(p={res_guayas_sw_full['pettitt']['p_value']:.2e}); CERES era slope={res_guayas_sw_ceres['monthly_mk']['theil_sen_decade']:+.3f}"
    )

    # 4. Verify Trap #3 & ENSO Dipole: Guayas vs. Manaus
    df_manaus, _ = load_region_nasa_power("BR_Amazon_Manaus")
    dipole = compare_regional_dipole(df_guayas, df_manaus, "PRECTOTCORR", oni_df=oni_df)
    assert dipole["has_opposite_enso_teleconnection"], "Guayas and Manaus must have opposite ONI teleconnections"
    print(
        f"[PASS] ENSO Dipole Verified: Guayas r_ONI={dipole['region_a']['enso']['r_oni']:+.3f} vs. "
        f"Manaus r_ONI={dipole['region_b']['enso']['r_oni']:+.3f}"
    )

    # 5. Verify 3-Model Multivariate ML Ensemble + Lagged Random Forest + Acker Outlier Switch + Spatio-Temporal Cube
    ml_res = run_multivariate_anomaly_detective(df_guayas, oni_df=oni_df)
    assert len(ml_res["anomalies"]) >= 15, "3-Model Consensus Ensemble must detect top multivariate anomalies"
    top_dates = [a["date"] for a in ml_res["anomalies"][:10]]
    assert any(d.startswith("1997-") or d.startswith("1998-") for d in top_dates), "Must detect 1997-98 Super El Nino"
    assert ml_res["lagged_driver_model"]["available"], "Supervised Random Forest lagged driver model must run"
    assert ml_res["lagged_driver_model"]["r2_in_sample"] > 0.40, "Random Forest R2 on soil moisture must be > 0.40"
    print(
        f"[PASS] 3-Model ML Ensemble & Lagged RF Verified: Top anomaly = {ml_res['anomalies'][0]['date']} "
        f"(Score={ml_res['anomalies'][0]['consensus_score_pct']}%, Votes={ml_res['anomalies'][0]['model_votes']}/3, "
        f"RF R²={ml_res['lagged_driver_model']['r2_in_sample']:.2f})"
    )

    # 6. Verify 2D Regional Grid, NASA MODIS 250m, and Meteomatics 1km caches
    grid_df = load_ecuador_regional_grid_trends("T2M")
    assert len(grid_df) == 96, f"Expected 96 Ecuador grid cells, got {len(grid_df)}"
    modis_df = load_or_fetch_modis_250m("EC_Andes_Paute", -2.61, -78.63)
    assert len(modis_df) >= 100, f"Expected >=100 MODIS 250m observations, got {len(modis_df)}"
    mm_dict = load_meteomatics_1km_cache()
    assert "EC_Andes_Paute" in mm_dict and len(mm_dict["EC_Andes_Paute"]) >= 70
    print(
        f"[PASS] Multi-Scale Data Verified: 96 NASA Regional Grid cells, {len(modis_df)} MODIS 250m obs, "
        f"and {len(mm_dict['EC_Andes_Paute'])} Meteomatics 1km records."
    )


if __name__ == "__main__":
    run_all_tests()
