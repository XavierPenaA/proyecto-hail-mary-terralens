/** Long-term trends view: monthly and annual series, climatology, six-variable scan, seasons. */
import { axis, baseOptions, renderChart } from '../charts.js';
import { escapeHtml, fmtNum, fmtP, pText } from '../format.js';
import { t } from '../i18n.js';
import {
  decadeUnits, monthLabel, regionName, seasonLabel, units, variableName, variableShort, verdictLabel, verdictTone, warningLabel,
} from '../labels.js';
import { chartColors } from '../theme.js';

const $ = (id) => document.getElementById(id);

function renderMonthlyChart(d) {
  const c = chartColors();
  const code = d.variable_info.code;
  const series = d.monthly_series;
  const anomalous = new Set(d.ml.anomalies.map((a) => a.date));
  const pet = d.pettitt;

  $('mainChartTitle').textContent = `${regionName(d.region.id)} — ${variableName(code)} (${d.year_start}–${d.year_end})`;
  renderChart(
    'monthlyTrendChart',
    {
      type: 'line',
      data: {
        labels: series.map((r) => r.date),
        datasets: [
          { label: t('legend.anomaly'), data: series.map((r) => r.anomaly), borderColor: c.anomaly, borderWidth: 1.2, pointRadius: 0, tension: 0, order: 4 },
          { label: t('legend.theilSen'), data: series.map((r) => r.ts_line), borderColor: c.trend, borderWidth: 2.2, pointRadius: 0, order: 1 },
          { label: t('legend.ols'), data: series.map((r) => r.ols_line), borderColor: c.ols, borderWidth: 1.6, borderDash: [5, 4], pointRadius: 0, order: 2 },
          {
            label: `${t('legend.pettitt')} (${pet.change_date})`,
            data: series.map((r) => (r.date <= pet.change_date ? pet.mean_before : pet.mean_after)),
            borderColor: c.step, borderWidth: 1.6, stepped: true, pointRadius: 0, order: 3,
          },
          {
            label: t('legend.mlExtreme'),
            data: series.map((r) => (anomalous.has(r.date) ? r.anomaly : null)),
            borderColor: c.alert, backgroundColor: c.alert, pointRadius: 3, showLine: false, order: 0,
          },
        ],
      },
      options: baseOptions({
        scales: {
          x: axis({ grid: false, ticks: { maxTicksLimit: 10 } }),
          y: axis({ title: `${t('axis.anomaly')} (${units(code)})` }),
        },
      }),
    },
    t('aria.monthlyChart', {
      region: regionName(d.region.id), variable: variableShort(code),
      slope: fmtNum(d.monthly_mk.theil_sen_decade, 3, true), units: decadeUnits(code),
    }),
  );
}

function renderAnnualChart(d) {
  const c = chartColors();
  const code = d.variable_info.code;
  renderChart(
    'annualTrendChart',
    {
      type: 'line',
      data: {
        labels: d.annual_series.map((r) => r.year),
        datasets: [
          { label: t('legend.annual'), data: d.annual_series.map((r) => r.annual_anomaly), borderColor: c.anomaly, borderWidth: 1.6, pointRadius: 2, tension: 0 },
          { label: t('legend.theilSen'), data: d.annual_series.map((r) => r.ts_line), borderColor: c.trend, borderWidth: 2, pointRadius: 0 },
        ],
      },
      options: baseOptions({
        scales: { x: axis({ grid: false, ticks: { maxTicksLimit: 8 } }), y: axis({ title: `${t('axis.anomaly')} (${units(code)})` }) },
      }),
    },
    t('aria.annualChart', { p: pText(d.annual_mk.p_original) }),
  );
}

function renderClimatology(d) {
  const c = chartColors();
  const clim = d.monthly_climatology;
  const band = (sign) => clim.map((m) => +(m.clim_mean + sign * m.clim_std).toFixed(4));
  renderChart(
    'climatologyChart',
    {
      type: 'line',
      data: {
        labels: clim.map((m) => monthLabel(m.month)),
        datasets: [
          { label: t('legend.normal'), data: clim.map((m) => m.clim_mean), borderColor: c.anomaly, borderWidth: 2, pointRadius: 3, tension: 0.3 },
          { label: '+1σ', data: band(1), borderColor: c.muted, borderWidth: 1, borderDash: [4, 3], pointRadius: 0, tension: 0.3 },
          { label: '−1σ', data: band(-1), borderColor: c.muted, borderWidth: 1, borderDash: [4, 3], pointRadius: 0, tension: 0.3 },
        ],
      },
      options: baseOptions({ scales: { x: axis({ grid: false }), y: axis({ title: units(d.variable_info.code) }) } }),
    },
    t('aria.climatology'),
  );
}

function verdictCell(code) {
  const tone = verdictTone(code);
  return `<span class="status-inline tone-${tone}"><span class="dot tone-${tone}"></span>${escapeHtml(verdictLabel(code))}</span>`;
}

function renderScan(d, onSelectVariable) {
  const active = d.variable_info.code;
  const tbody = $('fdrScanTableBody');
  tbody.replaceChildren();
  d.auto_scan.rows.forEach((r) => {
    const tr = document.createElement('tr');
    tr.tabIndex = 0;
    tr.style.cursor = 'pointer';
    if (r.variable === active) tr.setAttribute('aria-current', 'true');
    if (r.variable === active) tr.style.background = 'var(--accent-soft)';
    const warnings = r.warnings.length ? ` <span class="tone-warn" title="${escapeHtml(r.warnings.map((w) => warningLabel(w, { date: r.pettitt_date })).join(' '))}">⚑</span>` : '';
    tr.innerHTML = `
      <td style="color:var(--text-primary);font-weight:600;">${escapeHtml(variableShort(r.variable))} <span class="tabular" style="color:var(--text-tertiary);font-size:11px;">${r.period[0]}–${r.period[1]}</span></td>
      <td class="num" style="color:var(--text-primary);font-weight:600;">${fmtNum(r.theil_sen_decade, 4, true)} <span style="color:var(--text-tertiary);font-size:11px;">${escapeHtml(decadeUnits(r.variable))}</span></td>
      <td class="num expert-only">${fmtNum(r.ols_decade, 4, true)}</td>
      <td class="num expert-only">${fmtNum(r.lag1_r, 2)}</td>
      <td class="num expert-only">${fmtNum(r.n_eff, 0)}</td>
      <td class="num expert-only">${fmtP(r.p_naive)}</td>
      <td class="num">${fmtP(r.p_hamed_rao)}</td>
      <td class="num">${fmtP(r.q_fdr)}</td>
      <td class="num expert-only">${fmtP(r.p_annual_mk)}</td>
      <td class="tabular expert-only">${escapeHtml(r.pettitt_date || '—')} <span style="color:var(--text-tertiary);">(${pText(r.pettitt_p, { bootstrap: true })})</span></td>
      <td class="num expert-only">${fmtNum(r.r_oni, 2, true)}</td>
      <td>${verdictCell(r.verdict_code)}${warnings}</td>`;
    const select = () => onSelectVariable(r.variable);
    tr.addEventListener('click', select);
    tr.addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), select()));
    tbody.appendChild(tr);
  });

  const group = d.auto_scan.coincident_breaks;
  $('coincidentNote').hidden = !group;
  if (group) {
    $('coincidentNote').textContent = t('scan.coincident', {
      n: group.variables.length, date: group.date, variables: group.variables.map(variableShort).join(', '),
    });
  }
}

function renderSeasons(d) {
  const code = d.variable_info.code;
  $('seasonalTrendsBody').replaceChildren(
    ...d.seasonal_trends.map((s) => {
      const tr = document.createElement('tr');
      const tone = s.significant ? 'sig' : 'none';
      tr.innerHTML = `
        <td><span class="tabular" style="color:var(--accent-text);font-weight:700;">${s.season}</span> <span style="color:var(--text-secondary);">${escapeHtml(seasonLabel(s.season))}</span></td>
        <td class="num" style="color:var(--text-primary);font-weight:600;">${fmtNum(s.theil_sen_decade, 4, true)} <span style="color:var(--text-tertiary);font-size:11px;">${escapeHtml(decadeUnits(code))}</span></td>
        <td class="num expert-only">${fmtNum(s.ols_decade, 4, true)}</td>
        <td class="num">${fmtP(s.p_hamed_rao)}</td>
        <td><span class="status-inline tone-${tone}"><span class="dot tone-${tone}"></span>${escapeHtml(t(s.significant ? 'scan.significant' : 'scan.notSignificant'))}</span></td>`;
      return tr;
    }),
  );
}

export function renderTrends(d, { onSelectVariable }) {
  renderMonthlyChart(d);
  renderAnnualChart(d);
  renderClimatology(d);
  renderScan(d, onSelectVariable);
  renderSeasons(d);
}
