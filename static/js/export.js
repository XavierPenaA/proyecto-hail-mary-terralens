/** CSV download of the current monthly series, with the metadata needed to reuse it. */
import { fmtP } from './format.js';

const csvValue = (value) => (value === null || value === undefined ? '' : String(value));

export function exportSeriesCsv(d) {
  if (!d) return;
  const v = d.variable_info;
  const r = d.region;
  const metadata = [
    '# TerraLens export',
    `# region: ${r.id} (${r.name}), lat ${r.lat}, lon ${r.lon}`,
    `# variable: ${v.code} (${v.name}), units: ${v.units}`,
    `# source: ${v.source}`,
    `# period: ${d.year_start}-${d.year_end}, months analysed: ${d.n_months}`,
    '# anomaly = value minus the mean of the same calendar month; z_score = anomaly / std of that month',
    `# Theil-Sen trend: ${d.monthly_mk.theil_sen_decade.toFixed(4)} ${v.decade_units}; Hamed-Rao p: ${fmtP(d.monthly_mk.p_hamed_rao)}; verdict: ${d.verdict.code}`,
    `# trend warnings: ${d.warnings.join(', ') || 'none'}`,
    `# anomalous months removed: ${d.filters.n_excluded_months}`,
    `# generated: ${new Date().toISOString()}`,
    '# data: NASA POWER (https://power.larc.nasa.gov/), NASA Earthdata open data policy',
  ];
  const columns = [
    ['date', 'date'], ['value', 'raw_val'], ['climatology_mean', 'clim_mean'], ['anomaly', 'anomaly'],
    ['z_score', 'z_score'], ['ols_trend', 'ols_line'], ['theil_sen_trend', 'ts_line'],
  ];
  const lines = [
    ...metadata,
    columns.map(([name]) => name).join(','),
    ...d.monthly_series.map((row) => columns.map(([, key]) => csvValue(row[key])).join(',')),
  ];

  const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' }));
  const link = Object.assign(document.createElement('a'), {
    href: url,
    download: `TerraLens_${r.id}_${v.code}_${d.year_start}-${d.year_end}.csv`,
  });
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
