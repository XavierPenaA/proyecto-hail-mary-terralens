"""
Monte Carlo calibration of the TerraLens trend pipeline (runs the app's own functions).

Questions answered
1. Type I error: under AR(1) noise with NO trend, how often does each test (naive MK, Hamed-Rao,
   Santer OLS, and the final verdict) call a trend significant at alpha = 0.05?
2. Interval coverage: how often do the raw and the widened Theil-Sen 95% CI, and the Santer OLS CI,
   exclude zero under the same null?
3. Power: detection rate for a known trend (in units of noise sigma per decade).
4. Step-vs-trend: how often the BIC rule prefers 'step' on a series with no break, and how often it
   finds a real step of 1 or 2 sigma.

Usage: python scripts/calibration_study.py [--reps 300] [--out models/calibration_report.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.stats_engine import (  # noqa: E402
    SIGNIFICANCE_ALPHA, _pettitt_statistic, classify_trend_verdict, compare_step_vs_trend,
    mann_kendall_hamed_rao, ols_with_santer_correction, widen_theil_sen_ci,
)

N_MONTHS = 540
PHIS = (0.0, 0.5, 0.7, 0.85, 0.95)
TRENDS_SIGMA_PER_DECADE = (0.1, 0.2, 0.4)
STEP_SIZES_SIGMA = (1.0, 2.0)
SEED = 20261115


def ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance AR(1) series."""
    e = rng.standard_normal(n + 200) * np.sqrt(1 - phi ** 2)
    x = np.zeros(n + 200)
    for i in range(1, n + 200):
        x[i] = phi * x[i - 1] + e[i]
    return x[200:]


def run_trial(y: np.ndarray, t: np.ndarray) -> dict:
    ols = ols_with_santer_correction(t, y)
    raw_mk = mann_kendall_hamed_rao(t, y)
    mk = widen_theil_sen_ci(raw_mk, ols["n_eff"])
    verdict = classify_trend_verdict(mk["theil_sen_decade"], mk["p_hamed_rao"],
                                     p_santer=ols["p_santer"], lag1_r=ols["lag1_r"])
    return {
        "naive_mk": raw_mk["p_original"] < SIGNIFICANCE_ALPHA,
        "hamed_rao": raw_mk["p_hamed_rao"] < SIGNIFICANCE_ALPHA,
        "santer_ols": ols["p_santer"] < SIGNIFICANCE_ALPHA,
        "verdict": bool(verdict["is_significant"]),
        "ts_ci_raw_excludes_0": not (raw_mk["theil_sen_ci_low_decade"] <= 0 <= raw_mk["theil_sen_ci_high_decade"]),
        "ts_ci_widened_excludes_0": not (mk["theil_sen_ci_low_decade"] <= 0 <= mk["theil_sen_ci_high_decade"]),
        "santer_ci_excludes_0": not (ols["ci95_santer_decade_low"] <= 0 <= ols["ci95_santer_decade_high"]),
    }


def step_preference(y: np.ndarray, t: np.ndarray) -> str:
    """The app's own BIC rule, with the break at the Pettitt argmax."""
    k = int(np.argmax(_pettitt_statistic(y)[0]))
    return compare_step_vs_trend(t, y, k)["preferred_model"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--out", default="models/calibration_report.json")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    t = 1981 + np.arange(N_MONTHS) / 12.0
    report = {"n_months": N_MONTHS, "reps": args.reps, "alpha": SIGNIFICANCE_ALPHA,
              "type1": {}, "power": {}, "step_preference": {}}

    print(f"Type I error (no trend), {args.reps} reps per phi")
    for phi in PHIS:
        trials = [run_trial(ar1(N_MONTHS, phi, rng), t) for _ in range(args.reps)]
        rates = {key: round(float(np.mean([tr[key] for tr in trials])), 3) for key in trials[0]}
        report["type1"][str(phi)] = rates
        print(f"  phi={phi:<4} " + "  ".join(f"{k}={v:.3f}" for k, v in rates.items()))

    print("Power (AR(1) phi=0.7, trend in sigma/decade)")
    for slope in TRENDS_SIGMA_PER_DECADE:
        trials = [run_trial(ar1(N_MONTHS, 0.7, rng) + slope * (t - t[0]) / 10.0, t) for _ in range(args.reps)]
        rates = {key: round(float(np.mean([tr[key] for tr in trials])), 3) for key in ("hamed_rao", "santer_ols", "verdict")}
        report["power"][str(slope)] = rates
        print(f"  {slope} sigma/dec: " + "  ".join(f"{k}={v:.3f}" for k, v in rates.items()))

    print("BIC preference: false 'step' without a break, and step detection with a real jump")
    for phi in PHIS:
        row = {"false_step": round(float(np.mean(
            [step_preference(ar1(N_MONTHS, phi, rng), t) == "step" for _ in range(args.reps)])), 3)}
        for jump in STEP_SIZES_SIGMA:
            hits = []
            for _ in range(args.reps):
                y = ar1(N_MONTHS, phi, rng)
                y[N_MONTHS // 2:] += jump
                hits.append(step_preference(y, t) == "step")
            row[f"step_detected_{jump}sigma"] = round(float(np.mean(hits)), 3)
        report["step_preference"][str(phi)] = row
        print(f"  phi={phi:<4} " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()))

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
