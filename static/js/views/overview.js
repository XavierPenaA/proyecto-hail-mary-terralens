/** Plain-language answer card and the four key metric cards for the selected series. */
import { fmtNum, fmtP, pText } from '../format.js';
import { pick, t } from '../i18n.js';
import { decadeUnits, qualifiedVerdict, regionName, units, variableShort, verdictLabel, verdictTone, warningLabel } from '../labels.js';
import { token } from '../theme.js';

const $ = (id) => document.getElementById(id);

function answerSentence(d, code) {
  const slope = d.monthly_mk.theil_sen_decade;
  // "rises 0.2" / "falls 0.2": the verb carries the sign, except when the trend is rejected.
  const verbCarriesSign = code.startsWith('sig_') || code.startsWith('marginal_');
  const params = {
    region: regionName(d.region.id),
    variable: variableShort(d.variable_info.code).toLowerCase(),
    slope: verbCarriesSign ? fmtNum(Math.abs(slope), 3) : fmtNum(slope, 3, true),
    units: decadeUnits(d.variable_info.code),
    from: d.year_start,
    to: d.year_end,
    p: pText(d.monthly_mk.p_hamed_rao),
    direction: t(`direction.${d.verdict.direction}`),
  };
  return t(`answer.${code}`, params);
}

function warningBadges(d) {
  const params = {
    date: d.pettitt.change_date,
    splice: d.known_break?.break_date,
    delta: d.known_break ? fmtNum(d.known_break.step_delta, 2, true) : '',
    units: units(d.variable_info.code),
  };
  const badges = d.warnings.map((code) => warningLabel(code, params));
  if (d.window.start_moved_for_variable) badges.push(t('warning.start_moved', { year: d.year_start }));
  if (d.filters.exclude_ml_extremes) badges.push(t('warning.extremes_removed', { n: d.filters.n_excluded_months }));
  return badges;
}

function renderSparkline(values, label) {
  const svg = $('heroSparkline');
  svg.setAttribute('aria-label', label);
  if (!values || values.length < 3) {
    svg.replaceChildren();
    return;
  }
  const w = 140, h = 40, pad = 3;
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const pts = values.map((v, i) => `${(pad + (i / (values.length - 1)) * (w - 2 * pad)).toFixed(1)},${(h - pad - ((v - min) / span) * (h - 2 * pad)).toFixed(1)}`);
  const color = token('--series-anomaly');
  svg.innerHTML = `<polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />`;
}

export function renderSummary(d) {
  const code = d.variable_info.code;
  const verdictCode = qualifiedVerdict(d.verdict.code, d.warnings);
  const tone = verdictTone(verdictCode);

  $('answerKicker').textContent = `${regionName(d.region.id)} · ${variableShort(code)} · ${d.year_start}–${d.year_end}`;
  $('answerText').textContent = answerSentence(d, verdictCode);
  $('answerContext').textContent = `${pick(d.variable_info.explainer)} ${pick(d.region.summary)}`;
  $('answerWarnings').replaceChildren(
    ...warningBadges(d).map((text) => Object.assign(document.createElement('span'), { className: 'badge badge-warn', textContent: text })),
  );
  const verdict = $('answerVerdict');
  verdict.className = `badge tone-${tone}`;
  verdict.textContent = verdictLabel(verdictCode);
  renderSparkline(d.annual_series.map((r) => r.annual_anomaly), t('kpi.sparklineAria', { from: d.year_start, to: d.year_end }));

  // Trend
  $('kpiSlope').textContent = fmtNum(d.monthly_mk.theil_sen_decade, 3, true);
  $('kpiSlopeUnits').textContent = decadeUnits(code);
  $('kpiOlsCompare').textContent = `OLS ${fmtNum(d.monthly_ols.slope_decade, 3, true)}`;
  $('kpiWindowMeta').textContent = t('kpi.window', { from: d.year_start, to: d.year_end, n: d.n_months });

  // Memory and significance
  $('kpiLag1').textContent = `r₁ = ${fmtNum(d.monthly_ols.lag1_r, 2)}`;
  $('kpiPValue').textContent = fmtP(d.monthly_mk.p_hamed_rao);
  $('kpiNeff').textContent = t('kpi.neff', { neff: fmtNum(d.monthly_ols.n_eff, 0), n: d.n_months });
  $('kpiAnnualP').textContent = t('kpi.annualP', { p: pText(d.annual_mk.p_original) });

  // Largest shift and step-vs-trend check
  const pet = d.pettitt;
  $('kpiPettittDate').textContent = pet.change_date || '—';
  $('kpiPettittDelta').textContent = `Δ ${fmtNum(pet.step_delta, 2, true)} ${units(code)}`;
  $('kpiPettittP').textContent = pText(pet.p_value, { bootstrap: true });
  $('kpiShape').textContent = t(`shape.${d.homogeneity.preferred_model}`, { dbic: fmtNum(Math.abs(d.homogeneity.delta_bic), 1) });
  const kb = d.known_break;
  $('kpiKnownBreak').hidden = !kb;
  if (kb) {
    $('kpiKnownBreak').textContent = t('kpi.knownBreak', {
      date: kb.break_date, delta: fmtNum(kb.step_delta, 2, true), units: units(code), p: pText(kb.p_value, { bootstrap: true }),
    });
  }

  // Anomalous months and ENSO
  const ml = d.ml;
  $('kpiAnomalyCount').textContent = ml.anomalies.length;
  $('kpiAnomalyShare').textContent = ml.n_months ? `${fmtNum((100 * ml.anomalies.length) / ml.n_months, 1)}%` : '';
  $('kpiOniCorr').textContent = t('kpi.oni', {
    r: fmtNum(d.enso.r_oni, 2, true), p: pText(d.enso.p_oni), lag: d.enso.best_lag, rBest: fmtNum(d.enso.r_best, 2, true),
  });
}
