/** Model performance view, drawn from the offline benchmark manifest (train_models.py). */
import { axis, baseOptions, renderChart } from '../charts.js';
import { escapeHtml, fmtNum } from '../format.js';
import { t } from '../i18n.js';
import { featureLabel, regionName } from '../labels.js';
import { chartColors } from '../theme.js';

const $ = (id) => document.getElementById(id);

const r2Cell = (value) => `<span class="${value < 0 ? 'tone-bad' : ''}">${fmtNum(value, 3)}</span>`;
const gainCell = (value) => `<span class="${value > 0.01 ? 'tone-good' : value < -0.01 ? 'tone-bad' : 'tone-none'}">${fmtNum(value, 3, true)}</span>`;
const modelName = (name) => t(`model.${name}`);

function leaderboardRow(m, prefix = '') {
  return `
    ${prefix}
    <td style="color:var(--text-primary);font-weight:600;">${escapeHtml(modelName(m.model_name))}</td>
    <td class="num">${r2Cell(m.cv_r2_mean)} <span style="color:var(--text-tertiary);font-size:11px;">±${fmtNum(m.cv_r2_std, 3)}</span></td>
    <td class="num">${r2Cell(m.test_r2)}</td>
    <td class="num">${m.model_name === 'PersistenceBaseline' ? '—' : gainCell(m.test_r2_gain_vs_persistence)}</td>`;
}

export function renderSidebarBenchmark(manifest) {
  const board = manifest?.nowcast_benchmark?.leaderboard;
  if (!board) return;
  const best = board.find((m) => m.model_name === manifest.nowcast_benchmark.best_model);
  const persistence = board.find((m) => m.model_name === 'PersistenceBaseline');
  $('sbTrainedAt').textContent = (manifest.trained_at_utc || '').slice(0, 10);
  $('sbBestTest').textContent = best ? fmtNum(best.test_r2, 3) : '—';
  $('sbPersistence').textContent = persistence ? fmtNum(persistence.test_r2, 3) : '—';
}

export function renderTelemetry(manifest) {
  const nowcast = manifest?.nowcast_benchmark;
  if (!nowcast) {
    $('cvLeaderboardBody').innerHTML = `<tr><td colspan="6" class="empty-state">${escapeHtml(t('telemetry.missing'))}</td></tr>`;
    return;
  }
  const split = manifest.split;
  $('cvTrainedStamp').textContent = t('telemetry.stamp', {
    date: (manifest.trained_at_utc || '').slice(0, 10),
    dev: `${split.development[0]}–${split.development[1]}`,
    test: `${split.test[0]}–${split.test[1]}`,
  });

  $('cvLeaderboardBody').replaceChildren(
    ...nowcast.leaderboard.map((m) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `${leaderboardRow(m)}
        <td class="num expert-only">${fmtNum(m.test_rmse, 3)}σ</td>
        <td class="tabular expert-only" style="font-size:11.5px;color:var(--text-secondary);white-space:nowrap;">${m.fold_r2_scores.map((f) => fmtNum(f, 2)).join(' · ')}</td>`;
      return tr;
    }),
  );

  const c = chartColors();
  const perms = nowcast.permutation_importance.slice(0, 8);
  renderChart(
    'cubePermChart',
    {
      type: 'bar',
      data: {
        labels: perms.map((p) => featureLabel(p.feature)),
        datasets: [{ label: t('telemetry.perm.axis'), data: perms.map((p) => p.importance_mean), backgroundColor: c.trend, borderRadius: 5, borderSkipped: false }],
      },
      options: baseOptions({
        indexAxis: 'y',
        interaction: { mode: 'nearest', intersect: true },
        scales: { x: axis({ title: t('telemetry.perm.axis'), beginAtZero: true }), y: axis({ grid: false }) },
      }),
    },
    t('aria.perm', { feature: featureLabel(perms[0]?.feature) }),
  );

  const permBody = $('nowcastPermBody');
  if (permBody) {
    permBody.replaceChildren(
      ...nowcast.permutation_importance.map((p) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="color:var(--text-primary);font-weight:600;">${escapeHtml(featureLabel(p.feature))}</td>
          <td class="num tabular">${fmtNum(p.importance_mean, 4)}</td>`;
        return tr;
      }),
    );
  }

  $('forecastBody').replaceChildren(
    ...manifest.forecast_benchmarks.flatMap((bench) =>
      bench.leaderboard.map((m, i) => {
        const tr = document.createElement('tr');
        const horizon = i === 0 ? `<td rowspan="${bench.leaderboard.length}" class="tabular" style="font-weight:700;">t+${bench.horizon_months}</td>` : '';
        tr.innerHTML = leaderboardRow(m, horizon);
        return tr;
      }),
    ),
  );

  $('regionalTelemetryBody').replaceChildren(
    ...Object.values(manifest.regional_telemetry).map((r) => {
      const tr = document.createElement('tr');
      const top = r.top_drivers[0];
      tr.innerHTML = `
        <td style="color:var(--text-primary);font-weight:600;">${escapeHtml(regionName(r.region_id))}</td>
        <td class="num">${r2Cell(r.driver_cv_r2_mean)}</td>
        <td class="num">${r2Cell(r.driver_test_r2)}</td>
        <td class="num">${r2Cell(r.persistence_test_r2)}</td>
        <td class="num expert-only">${r.optimal_k}</td>
        <td class="num expert-only">${fmtNum(r.best_silhouette, 2)}</td>
        <td><code>${escapeHtml(top?.feature ?? '—')}</code></td>`;
      return tr;
    }),
  );
}
