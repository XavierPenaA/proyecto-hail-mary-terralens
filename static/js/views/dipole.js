/** Two-region comparison: anomaly cross-correlation, ENSO polarity and trend directions. */
import { getJson } from '../api.js';
import { axis, baseOptions, renderChart } from '../charts.js';
import { escapeHtml, fmtNum, pText, qText } from '../format.js';
import { t } from '../i18n.js';
import { decadeUnits, qualifiedVerdict, regionName, variableShort, verdictLabel, verdictTone } from '../labels.js';
import { state } from '../state.js';
import { chartColors } from '../theme.js';

const $ = (id) => document.getElementById(id);

function regionCard(stats, regionId, variable) {
  const code = qualifiedVerdict(stats.verdict.code, stats.warnings);
  const tone = verdictTone(code);
  return `
    <div class="kpi-card">
      <div class="kpi-label">${escapeHtml(regionName(regionId))} <span class="tabular">${stats.period.join('–')}</span></div>
      <div><span class="kpi-value tabular">${fmtNum(stats.theil_sen_decade, 3, true)}</span> <span class="kpi-unit">${escapeHtml(decadeUnits(variable))}</span></div>
      <div class="kpi-sub"><span class="tone-${tone}">${escapeHtml(verdictLabel(code))}</span> <span class="tabular">${pText(stats.p_hamed_rao)}</span></div>
      <div class="kpi-sub tabular">${escapeHtml(t('dipole.enso', { r: fmtNum(stats.enso.r_best, 2, true), lag: stats.enso.best_lag, q: qText(stats.enso.q_best) }))}</div>
    </div>`;
}

function render(d) {
  const variable = d.variable_info.code;
  const a = d.region_a_info.id;
  const b = d.region_b_info.id;
  const cross = d.cross_correlation;
  const strong = d.r_during_strong_enso === null ? '—' : fmtNum(d.r_during_strong_enso, 2, true);
  const trend = [d.stats_a, d.stats_b].some((stats) => qualifiedVerdict(stats.verdict.code, stats.warnings) === 'step_qualified')
    ? 'step_qualified' : d.trend_relation;

  $('dipoleSummaryStrip').innerHTML = `
    ${regionCard(d.stats_a, a, variable)}
    ${regionCard(d.stats_b, b, variable)}
    <div class="kpi-card">
      <div class="kpi-label">${escapeHtml(t('dipole.cross'))}</div>
      <div><span class="kpi-value tabular">${fmtNum(cross.r, 2, true)}</span> <span class="kpi-unit">r</span></div>
      <div class="kpi-sub tabular">${escapeHtml(t('dipole.crossDetail', { p: pText(cross.p), neff: fmtNum(cross.n_eff, 0), n: cross.n }))}</div>
      <div class="kpi-sub tabular">${escapeHtml(t('dipole.strongEnso', { r: strong, n: d.n_strong_enso_months }))}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">${escapeHtml(t('dipole.diagnosis'))}</div>
      <div style="font-size:14px;font-weight:600;color:var(--text-primary);">${escapeHtml(t(`relation.enso.${d.enso_relation}`))}</div>
      <div class="kpi-sub">${escapeHtml(t(`relation.trend.${trend}`))}</div>
    </div>`;

  const c = chartColors();
  renderChart(
    'dipoleChart',
    {
      type: 'line',
      data: {
        labels: d.series.map((r) => r.date),
        datasets: [
          { label: regionName(a), data: d.series.map((r) => r.z_a), borderColor: c.anomaly, borderWidth: 1.3, pointRadius: 0, tension: 0, yAxisID: 'y' },
          { label: regionName(b), data: d.series.map((r) => r.z_b), borderColor: c.alert, borderWidth: 1.3, pointRadius: 0, tension: 0, yAxisID: 'y' },
          { label: t('dipole.oni'), data: d.series.map((r) => r.oni), borderColor: c.oni, borderWidth: 1.4, borderDash: [4, 3], pointRadius: 0, yAxisID: 'yOni' },
        ],
      },
      options: baseOptions({
        legend: true,
        scales: {
          x: axis({ grid: false, ticks: { maxTicksLimit: 10 } }),
          y: axis({ title: t('dipole.axisZ', { variable: variableShort(variable) }) }),
          yOni: axis({ title: t('dipole.axisOni'), grid: false, position: 'right' }),
        },
      }),
    },
    t('aria.dipole', { a: regionName(a), b: regionName(b), r: fmtNum(cross.r, 2, true) }),
  );
}

export async function loadDipole() {
  const data = await getJson(
    '/api/dipole',
    {
      region_a: state.dipoleA, region_b: state.dipoleB, variable: state.dipoleVariable,
      start_year: state.startYear, end_year: state.endYear, homogeneous_ceres: state.homogeneous,
    },
    { key: 'dipole' },
  );
  $('dipoleResults').hidden = false;
  render(data);
  return data;
}
