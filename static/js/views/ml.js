/** Extreme-months view: PCA regimes, soil-moisture driver model and the anomaly table. */
import { axis, baseOptions, renderChart } from '../charts.js';
import { escapeHtml, fmtNum } from '../format.js';
import { t } from '../i18n.js';
import { eventLabel, featureLabel, oniPhaseLabel, regimeLabel, variableShort } from '../labels.js';
import { chartColors } from '../theme.js';

const $ = (id) => document.getElementById(id);

/** Axis label from the two variables with the largest loadings, e.g. "PC1: T2M +, RH2M −". */
function componentLabel(loadings, component) {
  const top = Object.entries(loadings)
    .sort(([, a], [, b]) => Math.abs(b[component]) - Math.abs(a[component]))
    .slice(0, 2)
    .map(([variable, l]) => `${variable}${l[component] >= 0 ? '+' : '−'}`);
  return `${component.toUpperCase()}: ${top.join(', ')}`;
}

function regimeNames(clusters) {
  const counts = {};
  return Object.fromEntries(
    clusters.map((c) => {
      counts[c.regime_code] = (counts[c.regime_code] || 0) + 1;
      const suffix = clusters.filter((o) => o.regime_code === c.regime_code).length > 1 ? ` ${counts[c.regime_code]}` : '';
      return [c.cluster_id, `${regimeLabel(c.regime_code)}${suffix}`];
    }),
  );
}

function renderPca(ml) {
  const c = chartColors();
  const palette = [c.anomaly, c.trend, c.ols, c.step];
  const names = regimeNames(ml.regimes.clusters);
  const [v1, v2] = ml.pca.variance_ratio;
  $('pcaSubtitle').textContent = t('ml.pca.variance', {
    pc1: fmtNum(v1 * 100, 1), pc2: fmtNum(v2 * 100, 1), k: ml.regimes.k,
  });

  const datasets = ml.regimes.clusters.map((cluster, i) => ({
    label: names[cluster.cluster_id],
    data: ml.pca_points.filter((p) => p.cluster_id === cluster.cluster_id && !p.is_anomaly).map((p) => ({ x: p.pca_1, y: p.pca_2, date: p.date })),
    backgroundColor: `${palette[i % palette.length]}88`,
    pointRadius: 2.5,
  }));
  datasets.push({
    label: t('legend.mlExtreme'),
    data: ml.pca_points.filter((p) => p.is_anomaly).map((p) => ({ x: p.pca_1, y: p.pca_2, date: p.date })),
    backgroundColor: c.alert,
    pointStyle: 'triangle',
    pointRadius: 5,
  });

  renderChart(
    'pcaScatterChart',
    {
      type: 'scatter',
      data: { datasets },
      options: baseOptions({
        legend: true,
        interaction: { mode: 'nearest', intersect: true },
        tooltip: { callbacks: { label: (ctx) => `${ctx.raw.date} · ${ctx.dataset.label}` } },
        scales: {
          x: axis({ title: componentLabel(ml.pca.loadings, 'pc1') }),
          y: axis({ title: componentLabel(ml.pca.loadings, 'pc2') }),
        },
      }),
    },
    t('aria.pca', { k: ml.regimes.k }),
  );

  $('clusterSummaryCards').replaceChildren(
    ...ml.regimes.clusters.map((cluster) => {
      const card = document.createElement('div');
      card.className = 'cluster-card';
      const z = cluster.mean_z;
      card.innerHTML = `
        <div style="font-weight:600;color:var(--text-primary);">${escapeHtml(names[cluster.cluster_id])}</div>
        <div class="tabular" style="color:var(--accent-text);">${t('ml.cluster.months', { n: cluster.count_months, pct: fmtNum(cluster.share_pct, 1) })}</div>
        <div class="tabular" style="color:var(--text-tertiary);">T2M ${fmtNum(z.T2M, 2, true)}σ · ${escapeHtml(variableShort('PRECTOTCORR'))} ${fmtNum(z.PRECTOTCORR, 2, true)}σ</div>`;
      return card;
    }),
  );
}

function renderDriverModel(ml) {
  const c = chartColors();
  const model = ml.lagged_driver_model;
  if (!model?.available) {
    $('rfBadge').textContent = '';
    $('rfPlainInsight').textContent = t('ml.rf.unavailable');
    renderChart('rfImportanceChart', { type: 'bar', data: { labels: [], datasets: [] }, options: baseOptions() });
    return;
  }
  $('rfBadge').textContent = t('ml.rf.badge', {
    cv: fmtNum(model.r2_cv_mean, 2), persistence: fmtNum(model.r2_cv_persistence, 2), n: model.n_samples,
  });
  const beats = model.r2_cv_mean > model.r2_cv_persistence;
  const top = model.top_features.slice(0, 8);
  $('rfPlainInsight').textContent = t(beats ? 'ml.rf.insightBeats' : 'ml.rf.insightLoses', {
    feature: featureLabel(top[0].feature_code), cv: fmtNum(model.r2_cv_mean, 2), persistence: fmtNum(model.r2_cv_persistence, 2),
  });
  renderChart(
    'rfImportanceChart',
    {
      type: 'bar',
      data: {
        labels: top.map((f) => featureLabel(f.feature_code)),
        datasets: [{ label: t('ml.rf.importance'), data: top.map((f) => f.importance_pct), backgroundColor: c.anomaly, borderRadius: 3 }],
      },
      options: baseOptions({
        indexAxis: 'y',
        interaction: { mode: 'nearest', intersect: true },
        scales: { x: axis({ title: t('ml.rf.importance'), beginAtZero: true }), y: axis({ grid: false }) },
      }),
    },
    t('aria.rf', { feature: featureLabel(top[0].feature_code) }),
  );
}

function renderAnomalyTable(ml) {
  const rows = ml.anomalies.slice(0, 20);
  const tbody = $('mlAnomaliesTableBody');
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-state">${escapeHtml(t('ml.anomalies.none'))}</td></tr>`;
    return;
  }
  tbody.replaceChildren(
    ...rows.map((a) => {
      const top = a.drivers.slice(0, 2).map((dr) => `${variableShort(dr.variable)} ${fmtNum(dr.z_sigma, 1, true)}σ`).join(' · ');
      const phase = a.oni_phase && a.oni_phase !== 'neutral' ? ` · ${oniPhaseLabel(a.oni_phase)}` : '';
      const detectors = Object.entries(a.detectors).filter(([, on]) => on).map(([name]) => t(`detector.${name}`)).join(', ');
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="tabular" style="color:var(--text-primary);font-weight:600;">${a.date}</td>
        <td class="num"><div class="prop-cell"><div class="prop-track"><div class="prop-fill" style="width:${Math.min(100, a.consensus_score_pct)}%;background:var(--series-alert);"></div></div><span>${fmtNum(a.consensus_score_pct, 0)}</span></div></td>
        <td class="num" title="${escapeHtml(detectors)}">${a.model_votes}/3</td>
        <td class="num expert-only">${fmtNum(a.mahalanobis_dist, 2)}</td>
        <td class="num">${a.oni === null ? '—' : fmtNum(a.oni, 2, true)}</td>
        <td><strong>${escapeHtml(eventLabel(a.event_code))}</strong><span style="color:var(--text-secondary);"> — ${escapeHtml(top)}${escapeHtml(phase)}</span></td>`;
      return tr;
    }),
  );
}

export function renderMl(d) {
  renderPca(d.ml);
  renderDriverModel(d.ml);
  renderAnomalyTable(d.ml);
}
