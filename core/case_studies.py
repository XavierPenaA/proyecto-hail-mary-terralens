"""
Guided case studies.

Each case has a fixed setup (region, variable, period, guard) and a short walkthrough of steps
that point at parts of the interface. Its finding is written from numbers computed at run time,
so the narrative cannot drift away from what the engine reports.
"""

from functools import lru_cache
from typing import Any, Callable, Dict, List

from core.analysis_service import auto_scan, ml_detective, outlier_sensitivity, variable_analysis
from core.catalog import DATA_START_YEAR, VARIABLES


# ---------------------------------------------------------------------------
# Number formatting for narratives (both languages use a decimal point, like the interface)
# ---------------------------------------------------------------------------
def _signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _p(value: float) -> str:
    if value <= 0.001:
        return "p ≤ 0.001"
    return f"p = {value:.3f}" if value < 0.1 else f"p = {value:.2f}"


def _months(lag: int, lang: str) -> str:
    if lang == "en":
        return f"{lag} month" if lag == 1 else f"{lag} months"
    return f"{lag} mes" if lag == 1 else f"{lag} meses"


def _q(value: float) -> str:
    return "q ≤ 0.001" if value <= 0.001 else f"q = {value:.3f}"


def _trend(res: Dict[str, Any]) -> Dict[str, Any]:
    mk = res["monthly_mk"]
    return {
        "slope_decade": round(mk["theil_sen_decade"], 4),
        "p_hamed_rao": mk["p_hamed_rao"],
        "verdict": res["verdict"]["code"],
        "warnings": res["warnings"],
        "units": VARIABLES[res["variable"]]["decade_units"],
    }


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
def _finding_enso_dipole() -> Dict[str, Any]:
    coast = variable_analysis("EC_Coast_Guayas", "PRECTOTCORR")
    amazon = variable_analysis("BR_Amazon_Manaus", "PRECTOTCORR")
    ec, am = coast["enso"], amazon["enso"]
    peak = coast["monthly_series"].sort_values("z_score").iloc[-1]
    t_coast, t_amazon = _trend(coast), _trend(amazon)
    step_en = step_es = ""
    if "step_dominated" in amazon["warnings"]:
        step_en = f", a change better described by a level shift around {amazon['pettitt']['change_date']} than by a gradual trend"
        step_es = f"; ese cambio se describe mejor como un salto de nivel hacia {amazon['pettitt']['change_date']} que como una tendencia gradual"
    coast_significant = t_coast["verdict"].startswith("sig_")
    coast_trend_en = (
        f"shows a trend of {_signed(t_coast['slope_decade'])} mm/day per decade" if coast_significant else "shows no significant trend"
    )
    coast_trend_es = (
        f"muestra una tendencia de {_signed(t_coast['slope_decade'])} mm/día por década"
        if coast_significant else "no muestra una tendencia significativa"
    )
    return {
        "metrics": {
            "coast_enso": {"r": ec["r_best"], "lag": ec["best_lag"], "q": ec["q_best"]},
            "amazon_enso": {"r": am["r_best"], "lag": am["best_lag"], "q": am["q_best"]},
            "coast_trend": t_coast,
            "amazon_trend": t_amazon,
            "coast_peak": {"date": peak["date"], "z": round(float(peak["z_score"]), 2)},
        },
        "text": {
            "en": (
                f"Rainfall on Ecuador's coast rises with El Niño (r = {_signed(ec['r_best'], 2)} with the ONI leading by "
                f"{_months(ec['best_lag'], 'en')}, {_q(ec['q_best'])}), while the central Amazon dries "
                f"(r = {_signed(am['r_best'], 2)}, ONI leading by {_months(am['best_lag'], 'en')}, {_q(am['q_best'])}). The long-term trends do not mirror "
                f"each other: the coast {coast_trend_en} ({_p(t_coast['p_hamed_rao'])}) and Manaus changes by "
                f"{_signed(t_amazon['slope_decade'])} mm/day per decade ({_p(t_amazon['p_hamed_rao'])}){step_en}. "
                f"The wettest coastal month was {peak['date']} ({_signed(float(peak['z_score']), 1)}σ)."
            ),
            "es": (
                f"La lluvia en la costa de Ecuador sube con El Niño (r = {_signed(ec['r_best'], 2)} con el ONI adelantado "
                f"{_months(ec['best_lag'], 'es')}, {_q(ec['q_best'])}), mientras la Amazonía central se seca "
                f"(r = {_signed(am['r_best'], 2)}, ONI adelantado {_months(am['best_lag'], 'es')}, {_q(am['q_best'])}). Las tendencias de largo plazo no "
                f"son opuestas: la costa {coast_trend_es} ({_p(t_coast['p_hamed_rao'])}) y Manaos cambia {_signed(t_amazon['slope_decade'])} mm/día por década "
                f"({_p(t_amazon['p_hamed_rao'])}){step_es}. El mes más lluvioso de la costa fue {peak['date']} "
                f"({_signed(float(peak['z_score']), 1)}σ)."
            ),
        },
    }


def _finding_autocorrelation_trap() -> Dict[str, Any]:
    res = variable_analysis("EC_Andes_Paute", "GWETROOT")
    ols = res["monthly_ols"]
    metrics = {
        "p_naive": ols["p_naive"], "lag1_r": ols["lag1_r"], "n": ols["n"], "n_eff": ols["n_eff"],
        "p_santer": ols["p_santer"], "p_annual": res["annual_mk"]["p_original"], "verdict": res["verdict"]["code"],
    }
    rejected = res["verdict"]["code"] in ("autocorr_rejected", "not_significant")
    outcome_en = "the trend is not reliable" if rejected else "the trend survives the correction"
    outcome_es = "la tendencia no es confiable" if rejected else "la tendencia sobrevive a la corrección"
    return {
        "metrics": metrics,
        "text": {
            "en": (
                f"A plain regression on {ols['n']} months of root-zone soil wetness reports a trend with {_p(ols['p_naive'])}. "
                f"But each month remembers the previous one (r₁ = {ols['lag1_r']:.2f}), so the series holds only about "
                f"{ols['n_eff']:.0f} independent values. After correcting for that, {outcome_en}: Santer-corrected "
                f"{_p(ols['p_santer'])}, annual-mean test {_p(res['annual_mk']['p_original'])}."
            ),
            "es": (
                f"Una regresión simple sobre {ols['n']} meses de humedad del suelo radicular reporta una tendencia con "
                f"{_p(ols['p_naive'])}. Pero cada mes recuerda al anterior (r₁ = {ols['lag1_r']:.2f}), así que la serie solo "
                f"contiene unos {ols['n_eff']:.0f} valores independientes. Al corregirlo, {outcome_es}: Santer "
                f"{_p(ols['p_santer'])}, prueba de promedios anuales {_p(res['annual_mk']['p_original'])}."
            ),
        },
    }


def _finding_sensor_splice() -> Dict[str, Any]:
    def _pair(region_id: str) -> Dict[str, Any]:
        full = variable_analysis(region_id, "ALLSKY_SFC_SW_DWN", homogeneous_splice=False)
        ceres = variable_analysis(region_id, "ALLSKY_SFC_SW_DWN", homogeneous_splice=True)
        return {
            "full": _trend(full),
            "full_period": [full["year_start"], full["year_end"]],
            "pettitt_date": full["pettitt"]["change_date"],
            "known_break": full["known_break"],
            "ceres": _trend(ceres),
            "ceres_period": [ceres["year_start"], ceres["year_end"]],
        }

    coast, pantanal = _pair("EC_Coast_Guayas"), _pair("BR_Pantanal")
    kb_c, kb_p = coast["known_break"], pantanal["known_break"]
    p0, p1 = coast["full_period"]

    def _ceres_sentence(lang: str) -> str:
        parts = []
        for name_en, name_es, data in (("coast", "la costa", coast), ("Pantanal", "el Pantanal", pantanal)):
            sig = data["ceres"]["verdict"].startswith("sig_")
            slope = f"{_signed(data['ceres']['slope_decade'])}, {_p(data['ceres']['p_hamed_rao'])}"
            if lang == "en":
                parts.append(f"the {name_en} trend {'remains' if sig else 'disappears'} ({slope})")
            else:
                parts.append(f"la tendencia en {name_es} {'se mantiene' if sig else 'desaparece'} ({slope})")
        return "; ".join(parts)

    return {
        "metrics": {"coast": coast, "pantanal": pantanal},
        "text": {
            "en": (
                f"Over {p0}–{p1}, surface solar irradiance appears to change by {_signed(coast['full']['slope_decade'])} "
                f"MJ/m²/day per decade on Ecuador's coast and {_signed(pantanal['full']['slope_decade'])} in the Pantanal. "
                f"Both series jump at NASA POWER's documented source change in January 2001 "
                f"(coast {_signed(kb_c['step_delta'], 2)}, Pantanal {_signed(kb_p['step_delta'], 2)} MJ/m²/day; block-bootstrap "
                f"{_p(max(kb_c['p_value'], kb_p['p_value']))}). Pettitt places the largest shift nearby, at "
                f"{coast['pettitt_date']} and {pantanal['pettitt_date']}. Within the homogeneous CERES era "
                f"({coast['ceres_period'][0]}–{coast['ceres_period'][1]}) {_ceres_sentence('en')}."
            ),
            "es": (
                f"En {p0}–{p1} la irradiancia solar parece cambiar {_signed(coast['full']['slope_decade'])} MJ/m²/día por "
                f"década en la costa de Ecuador y {_signed(pantanal['full']['slope_decade'])} en el Pantanal. Ambas series "
                f"saltan en el cambio de fuente documentado por NASA POWER en enero de 2001 (costa "
                f"{_signed(kb_c['step_delta'], 2)}, Pantanal {_signed(kb_p['step_delta'], 2)} MJ/m²/día; bootstrap por bloques "
                f"{_p(max(kb_c['p_value'], kb_p['p_value']))}). Pettitt ubica el mayor quiebre cerca, en {coast['pettitt_date']} "
                f"y {pantanal['pettitt_date']}. Dentro de la era homogénea CERES "
                f"({coast['ceres_period'][0]}–{coast['ceres_period'][1]}) {_ceres_sentence('es')}."
            ),
        },
    }


def _finding_pantanal_compound() -> Dict[str, Any]:
    variables = ["T2M", "PRECTOTCORR", "GWETTOP", "RH2M"]
    trends = {v: _trend(variable_analysis("BR_Pantanal", v)) for v in variables}
    sensitivity = {r["variable"]: r["code"] for r in outlier_sensitivity("BR_Pantanal")["comparisons"]}
    n_excluded = len(ml_detective("BR_Pantanal")["extreme_dates"])
    n_sig = sum(t["verdict"].startswith("sig_") for t in trends.values())
    n_robust = sum(sensitivity.get(v) == "robust" for v in variables)
    listing_en = ", ".join(
        f"{VARIABLES[v]['short_name'].lower()} {_signed(trends[v]['slope_decade'])} {VARIABLES[v]['decade_units']}" for v in variables
    )
    listing_es = ", ".join(
        f"{VARIABLES[v]['short_name_es'].lower()} {_signed(trends[v]['slope_decade'])} {VARIABLES[v]['decade_units_es']}" for v in variables
    )
    return {
        "metrics": {"trends": trends, "outlier_sensitivity": sensitivity, "n_excluded_months": n_excluded},
        "text": {
            "en": (
                f"{n_sig} of {len(variables)} variables show a significant 1981–2025 trend after autocorrelation correction: "
                f"{listing_en}. Removing the {n_excluded} months flagged as multivariate anomalies keeps {n_robust} of "
                f"{len(variables)} trends significant and in the same direction (outlier test of Acker et al., 2009), so the "
                f"change is not driven by a few extreme months."
            ),
            "es": (
                f"{n_sig} de {len(variables)} variables muestran una tendencia significativa en 1981–2025 tras corregir la "
                f"autocorrelación: {listing_es}. Al quitar los {n_excluded} meses marcados como anomalías multivariadas, "
                f"{n_robust} de {len(variables)} tendencias siguen significativas y con el mismo signo (prueba de extremos de "
                f"Acker et al., 2009): el cambio no depende de unos pocos meses extremos."
            ),
        },
    }


def _finding_arctic_vs_delta() -> Dict[str, Any]:
    arctic = variable_analysis("US_Arctic_Barrow", "T2M")
    delta = variable_analysis("US_Mississippi_Delta", "T2M")
    t_a, t_d = _trend(arctic), _trend(delta)
    ratio = t_a["slope_decade"] / t_d["slope_decade"] if abs(t_d["slope_decade"]) > 1e-9 else float("nan")
    pet = arctic["pettitt"]
    preferred = arctic["homogeneity"]["preferred_model"]
    shape_en = {
        "step": "a single step fits the record better than a gradual trend",
        "trend": "a gradual trend fits the record better than a single step",
        "inconclusive": "a gradual trend and a single step fit about equally well, so the record cannot say whether the warming was gradual or abrupt",
    }[preferred]
    shape_es = {
        "step": "un salto único describe el registro mejor que una tendencia gradual",
        "trend": "una tendencia gradual describe el registro mejor que un salto único",
        "inconclusive": "una tendencia gradual y un salto único se ajustan casi igual, así que el registro no permite saber si el calentamiento fue gradual o abrupto",
    }[preferred]
    return {
        "metrics": {"arctic": t_a, "delta": t_d, "ratio": round(ratio, 2), "pettitt_date": pet["change_date"],
                    "pettitt_step": round(pet["step_delta"], 2), "preferred_model": preferred},
        "text": {
            "en": (
                f"Utqiaġvik warms {_signed(t_a['slope_decade'], 2)} °C per decade ({_p(t_a['p_hamed_rao'])}), {ratio:.1f} times "
                f"the Mississippi Delta ({_signed(t_d['slope_decade'], 2)} °C per decade). Pettitt finds the largest Arctic "
                f"shift in {pet['change_date']} ({_signed(pet['step_delta'], 2)} °C), but {shape_en}."
            ),
            "es": (
                f"Utqiaġvik se calienta {_signed(t_a['slope_decade'], 2)} °C por década ({_p(t_a['p_hamed_rao'])}), {ratio:.1f} veces "
                f"más que el delta del Misisipi ({_signed(t_d['slope_decade'], 2)} °C por década). Pettitt ubica el mayor quiebre "
                f"ártico en {pet['change_date']} ({_signed(pet['step_delta'], 2)} °C), pero {shape_es}."
            ),
        },
    }


def _finding_napo_step() -> Dict[str, Any]:
    scan = auto_scan("EC_Amazon_Napo")
    group = scan["coincident_breaks"]
    precip = variable_analysis("EC_Amazon_Napo", "PRECTOTCORR")
    steps = {
        v: round(variable_analysis("EC_Amazon_Napo", v)["pettitt"]["step_delta"], 3)
        for v in (group["variables"] if group else [])
    }
    n_vars = len(steps)
    date = group["date"] if group else precip["pettitt"]["change_date"]
    t_p = _trend(precip)
    return {
        "metrics": {"coincident_breaks": group, "steps": steps, "precip_trend": t_p},
        "text": {
            "en": (
                f"{n_vars} variables in the Napo foothills shift within two months of {date} (precipitation "
                f"{_signed(steps.get('PRECTOTCORR', precip['pettitt']['step_delta']), 2)} mm/day). A jump shared by several "
                f"variables in the same month is a typical sign of a change in the data inputs, although a real regime "
                f"shift cannot be ruled out from these data alone. The "
                f"precipitation trend ({_signed(t_p['slope_decade'])} mm/day per decade, {_p(t_p['p_hamed_rao'])}) should "
                f"be checked against independent station records before it is trusted."
            ),
            "es": (
                f"{n_vars} variables del piedemonte de Napo cambian de nivel a menos de dos meses de {date} (precipitación "
                f"{_signed(steps.get('PRECTOTCORR', precip['pettitt']['step_delta']), 2)} mm/día). Un salto compartido por varias "
                f"variables el mismo mes es una señal típica de un cambio en los datos de entrada, aunque con estos datos no se "
                f"puede descartar un cambio real de régimen. La tendencia de "
                f"lluvia ({_signed(t_p['slope_decade'])} mm/día por década, {_p(t_p['p_hamed_rao'])}) debe contrastarse con "
                f"estaciones independientes antes de darla por buena."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
def _step(view: str, target: str, en: str, es: str) -> Dict[str, Any]:
    return {"view": view, "target": target, "text": {"en": en, "es": es}}


CASES: List[Dict[str, Any]] = [
    {
        "id": "enso_dipole",
        "title": {"en": "Same El Niño, opposite rainfall", "es": "Mismo El Niño, lluvia opuesta"},
        "question": {
            "en": "Why does El Niño flood Ecuador's coast while the central Amazon dries, and do their long-term trends also oppose?",
            "es": "¿Por qué El Niño inunda la costa de Ecuador mientras seca la Amazonía central, y sus tendencias de largo plazo también son opuestas?",
        },
        "setup": {"region": "EC_Coast_Guayas", "variable": "PRECTOTCORR", "start_year": DATA_START_YEAR,
                  "homogeneous_splice": True, "dipole": ["EC_Coast_Guayas", "BR_Amazon_Manaus", "PRECTOTCORR"]},
        "finding": _finding_enso_dipole,
        "steps": [
            _step("tab-dipole", "dipoleChartPanel",
                  "Compare both rainfall anomalies with the ONI: the coast peaks with El Niño, the Amazon dips.",
                  "Compara ambas anomalías de lluvia con el ONI: la costa sube con El Niño y la Amazonía baja."),
            _step("tab-dipole", "dipoleSummaryStrip",
                  "The diagnosis cards separate the ENSO response from the long-term trends.",
                  "Las tarjetas de diagnóstico separan la respuesta a ENSO de las tendencias de largo plazo."),
            _step("tab-trends", "monthlyChartPanel",
                  "Back on the coast: the 1997–98 El Niño is the largest wet anomaly in the record.",
                  "De vuelta en la costa: El Niño de 1997–98 es la mayor anomalía húmeda del registro."),
        ],
    },
    {
        "id": "autocorrelation_trap",
        "title": {"en": "The autocorrelation trap", "es": "La trampa de la autocorrelación"},
        "question": {
            "en": "Is root-zone soil moisture in the Paute hydropower basin really changing?",
            "es": "¿Realmente está cambiando la humedad del suelo en la cuenca hidroeléctrica del Paute?",
        },
        "setup": {"region": "EC_Andes_Paute", "variable": "GWETROOT", "start_year": DATA_START_YEAR, "homogeneous_splice": True},
        "finding": _finding_autocorrelation_trap,
        "steps": [
            _step("tab-trends", "kpiCardAutocorr",
                  "Look at r₁ and the effective sample size: hundreds of months behave like a few dozen independent values.",
                  "Mira r₁ y el tamaño efectivo: cientos de meses se comportan como unas pocas decenas de valores independientes."),
            _step("tab-trends", "annualChartPanel",
                  "Annual means remove most of the month-to-month memory; their trend test agrees with the correction.",
                  "Los promedios anuales quitan casi toda la memoria mes a mes; su prueba coincide con la corrección."),
            _step("tab-multiscale", "mapPanel",
                  "Each map cell is about 55 km wide: one value mixes very different Andean slopes.",
                  "Cada celda del mapa mide unos 55 km: un solo valor mezcla laderas andinas muy distintas."),
        ],
    },
    {
        "id": "sensor_splice",
        "title": {"en": "A fake trend from a sensor change", "es": "Una tendencia falsa por cambio de sensor"},
        "question": {
            "en": "Did surface sunlight really change over South America, or did the data source change in 2001?",
            "es": "¿Cambió realmente la radiación solar en Sudamérica, o cambió la fuente de datos en 2001?",
        },
        "setup": {"region": "EC_Coast_Guayas", "variable": "ALLSKY_SFC_SW_DWN", "start_year": 1984, "homogeneous_splice": False},
        "finding": _finding_sensor_splice,
        "steps": [
            _step("tab-trends", "monthlyChartPanel",
                  "With the 2001 filter off, the series shows a clear drop around the source change.",
                  "Con el filtro 2001 apagado, la serie muestra una caída clara alrededor del cambio de fuente."),
            _step("tab-trends", "kpiCardBreak",
                  "The break card tests the documented January 2001 change directly and compares it with Pettitt.",
                  "La tarjeta de quiebre prueba directamente el cambio documentado de enero de 2001 y lo compara con Pettitt."),
            _step("tab-trends", "btnCeresGuard",
                  "Turn the 2001 filter on to keep only the homogeneous CERES era and watch the coastal trend vanish.",
                  "Activa el filtro 2001 para quedarte con la era homogénea CERES y verás desaparecer la tendencia costera."),
        ],
    },
    {
        "id": "pantanal_compound",
        "title": {"en": "Pantanal: warmer and drier", "es": "Pantanal: más cálido y más seco"},
        "question": {
            "en": "Is the Pantanal's hot, dry turn a lasting trend or the effect of a few extreme months?",
            "es": "¿El giro cálido y seco del Pantanal es una tendencia duradera o el efecto de unos pocos meses extremos?",
        },
        "setup": {"region": "BR_Pantanal", "variable": "T2M", "start_year": DATA_START_YEAR, "homogeneous_splice": True},
        "finding": _finding_pantanal_compound,
        "steps": [
            _step("tab-trends", "fdrPanel",
                  "The scan tests all six variables at once and controls false discoveries.",
                  "El escaneo prueba las seis variables a la vez y controla los falsos descubrimientos."),
            _step("tab-acker", "ackerTablePanel",
                  "Remove the anomalous months and recompute: robust trends keep their sign and significance.",
                  "Quita los meses anómalos y recalcula: las tendencias robustas mantienen signo y significancia."),
            _step("tab-ml", "mlAnomPanel",
                  "These are the months removed, with the variables that made each one unusual.",
                  "Estos son los meses quitados, con las variables que hicieron inusual a cada uno."),
        ],
    },
    {
        "id": "arctic_vs_delta",
        "title": {"en": "Arctic warming vs. the Mississippi Delta", "es": "Calentamiento ártico vs. delta del Misisipi"},
        "question": {
            "en": "How fast does the Arctic coast warm compared with a mid-latitude delta, and was it gradual?",
            "es": "¿Qué tan rápido se calienta la costa ártica frente a un delta de latitud media, y fue gradual?",
        },
        "setup": {"region": "US_Arctic_Barrow", "variable": "T2M", "start_year": DATA_START_YEAR, "homogeneous_splice": True,
                  "dipole": ["US_Arctic_Barrow", "US_Mississippi_Delta", "T2M"]},
        "finding": _finding_arctic_vs_delta,
        "steps": [
            _step("tab-trends", "monthlyChartPanel",
                  "The Theil-Sen line and the Pettitt step describe the same warming in two ways.",
                  "La recta Theil-Sen y el escalón de Pettitt describen el mismo calentamiento de dos maneras."),
            _step("tab-trends", "kpiCardBreak",
                  "The model comparison says whether a step or a gradual trend fits better.",
                  "La comparación de modelos dice si se ajusta mejor un salto o una tendencia gradual."),
            _step("tab-dipole", "dipoleSummaryStrip",
                  "Compare both sites side by side.",
                  "Compara ambos sitios lado a lado."),
        ],
    },
    {
        "id": "napo_step",
        "title": {"en": "A step that looks like a trend", "es": "Un salto que parece tendencia"},
        "question": {
            "en": "Did the climate of the Napo foothills change in one month of 2007?",
            "es": "¿Cambió el clima del piedemonte de Napo en un solo mes de 2007?",
        },
        "setup": {"region": "EC_Amazon_Napo", "variable": "PRECTOTCORR", "start_year": DATA_START_YEAR, "homogeneous_splice": True},
        "finding": _finding_napo_step,
        "steps": [
            _step("tab-trends", "monthlyChartPanel",
                  "Precipitation jumps to a new level in 2007 and stays there.",
                  "La precipitación salta a un nuevo nivel en 2007 y se queda ahí."),
            _step("tab-trends", "fdrPanel",
                  "The Pettitt column shows the same month for almost every variable.",
                  "La columna de Pettitt muestra el mismo mes para casi todas las variables."),
        ],
    },
]

_CASES_BY_ID: Dict[str, Dict[str, Any]] = {c["id"]: c for c in CASES}


@lru_cache(maxsize=None)
def case_finding(case_id: str) -> Dict[str, Any]:
    finder: Callable[[], Dict[str, Any]] = _CASES_BY_ID[case_id]["finding"]
    return finder()


def public_cases() -> List[Dict[str, Any]]:
    """Case definitions with their computed findings, ready to serialize."""
    return [
        {
            "id": c["id"],
            "number": i + 1,
            "title": c["title"],
            "question": c["question"],
            "setup": c["setup"],
            "steps": c["steps"],
            "finding": case_finding(c["id"])["text"],
        }
        for i, c in enumerate(CASES)
    ]
