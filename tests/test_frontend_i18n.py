"""Both locale files hold the same keys and placeholders, and every key the interface uses exists."""

import json
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"
LOCALES = {lang: json.loads((STATIC / "i18n" / f"{lang}.json").read_text(encoding="utf-8")) for lang in ("en", "es")}

# Families of keys built at run time from API codes (t(`verdict.${code}`) and similar).
DYNAMIC_FAMILIES = {
    "verdict": ["sig_increase", "sig_decrease", "marginal_increase", "marginal_decrease", "not_significant", "autocorr_rejected"],
    "answer": ["sig_increase", "sig_decrease", "marginal_increase", "marginal_decrease", "not_significant", "autocorr_rejected", "step_qualified"],
    "direction": ["increase", "decrease"],
    "warning": ["step_dominated", "known_splice_in_window"],
    "shape": ["step", "trend", "inconclusive"],
    "event": ["warm_wet", "wet_extreme", "hot_dry", "dry_extreme", "cold_extreme", "multivariate"],
    "oni": ["strong_el_nino", "el_nino", "neutral", "la_nina", "strong_la_nina"],
    "regime": ["warm_dry", "wet", "dry", "warm", "cool", "near_normal"],
    "sensitivity": ["robust", "outlier_driven", "masked", "no_trend"],
    "zone": ["ecuador", "south_america", "global_reference"],
    "season": ["DJF", "MAM", "JJA", "SON"],
    "month": [str(m) for m in range(1, 13)],
    "feature": ["PREC", "T2M", "RH2M", "SW", "ONI"],
    "detector": ["isolation_forest", "lof", "mahalanobis"],
    "relation.enso": ["opposite", "same_sign", "not_both_significant"],
    "relation.trend": ["opposite", "same_direction", "not_both_significant", "step_qualified"],
    "model": ["PersistenceBaseline", "RidgeRegression", "RandomForestRegressor", "HistGradientBoostingRegressor"],
    "viewGuide": ["tab-trends", "tab-ml", "tab-acker", "tab-dipole", "tab-multiscale", "tab-telemetry", "tab-guide", "tab-about"],
}


def placeholders(text: str) -> set:
    return set(re.findall(r"\{(\w+)\}", text))


def test_locales_have_the_same_keys_and_placeholders():
    en, es = LOCALES["en"], LOCALES["es"]
    assert set(en) == set(es)
    assert [k for k in en if placeholders(en[k]) != placeholders(es[k])] == []


def test_static_markup_keys_exist():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    keys = set(re.findall(r'data-i18n(?:-title|-aria-label)?="([^"]+)"', html))
    assert keys and keys <= set(LOCALES["en"])


def test_script_keys_exist():
    used = set()
    for path in (STATIC / "js").rglob("*.js"):
        source = path.read_text(encoding="utf-8")
        used |= set(re.findall(r"\bt\(\s*'([^'$]+)'", source))
        # Keys chosen by a ternary, e.g. t(isLight() ? 'ctrl.theme.dark' : 'ctrl.theme.light')
        for first, second in re.findall(r"\?\s*'([a-z]+\.[\w.]+)'\s*:\s*'([a-z]+\.[\w.]+)'", source):
            used |= {first, second}
    used |= {f"{family}.{code}" for family, codes in DYNAMIC_FAMILIES.items() for code in codes}
    missing = sorted(k for k in used if k not in LOCALES["en"])
    assert missing == []
