/** Outlier sensitivity view: each variable's trend with and without the anomalous months. */
import { axis, baseOptions, renderChart } from '../charts.js';
import { escapeHtml, fmtNum, fmtP, fmtPct } from '../format.js';
import { t } from '../i18n.js';
import { decadeUnits, sensitivityLabel, variableShort } from '../labels.js';
import { chartColors } from '../theme.js';

const $ = (id) => document.getElementById(id);
const SENSITIVITY_TONE = { robust: 'good', outlier_driven: 'warn', masked: 'warn', no_trend: 'none' };

export function renderAcker(d) {
  const exp = d.outlier_sensitivity;
  const c = chartColors();
  $('ackerSubtitle').textContent = t('acker.subtitle', { n: exp.n_excluded_months });

  $('ackerTableBody').replaceChildren(
    ...exp.comparisons.map((r) => {
      const tone = SENSITIVITY_TONE[r.code] || 'none';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="color:var(--text-primary);font-weight:600;">${escapeHtml(variableShort(r.variable))} <span class="tabular" style="color:var(--text-tertiary);font-size:11px;">${r.period[0]}–${r.period[1]}</span></td>
        <td class="num">${fmtNum(r.slope_all_months, 4, true)} <span style="color:var(--text-tertiary);font-size:11px;">${escapeHtml(decadeUnits(r.variable))}</span></td>
        <td class="num expert-only">${fmtP(r.p_hr_all_months)}</td>
        <td class="num" style="color:var(--text-primary);font-weight:600;">${fmtNum(r.slope_without_extremes, 4, true)}</td>
        <td class="num expert-only">${fmtP(r.p_hr_without_extremes)}</td>
        <td class="num">${fmtPct(r.slope_delta_pct)}</td>
        <td><span class="tone-${tone}">${escapeHtml(sensitivityLabel(r.code))}</span></td>`;
      return tr;
    }),
  );

  renderChart(
    'ackerBarChart',
    {
      type: 'bar',
      data: {
        labels: exp.comparisons.map((r) => variableShort(r.variable)),
        datasets: [
          { label: t('acker.allMonths'), data: exp.comparisons.map((r) => r.slope_all_months), backgroundColor: c.anomaly, borderRadius: 3 },
          { label: t('acker.withoutExtremes'), data: exp.comparisons.map((r) => r.slope_without_extremes), backgroundColor: c.trend, borderRadius: 3 },
        ],
      },
      options: baseOptions({
        indexAxis: 'y',
        legend: true,
        interaction: { mode: 'nearest', intersect: true },
        scales: { x: axis({ title: t('acker.axis') }), y: axis({ grid: false }) },
      }),
    },
    t('aria.acker', { n: exp.n_excluded_months }),
  );
}
