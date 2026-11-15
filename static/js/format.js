/** Number formatting shared by every view (decimal point in both languages, like the API). */
const MINUS = '−';

export function fmtNum(value, digits = 3, signed = false) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  const n = Number(value);
  const text = Math.abs(n).toFixed(digits);
  if (n < 0 && Number(text) !== 0) return `${MINUS}${text}`;
  return signed && n > 0 ? `+${text}` : text;
}

/** p-values: '< 0.0001' for tiny values, '≤ 0.001' for the bootstrap floor (1 / 1000). */
export function fmtP(value, { bootstrap = false } = {}) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  const n = Number(value);
  if (bootstrap && n <= 0.001) return '≤ 0.001';
  if (n < 0.0001) return '< 0.0001';
  return n < 0.01 ? n.toFixed(4) : n.toFixed(3);
}

/** p-value as a phrase: 'p = 0.012', 'p < 0.0001' or 'p ≤ 0.001'. */
export function pText(value, options) {
  const text = fmtP(value, options);
  return /^[<≤]/.test(text) ? `p ${text}` : `p = ${text}`;
}

/** q-value (false discovery rate) as a phrase, like pText. */
export function qText(value) {
  return pText(value).replace(/^p/, 'q');
}

export const fmtPct = (value, digits = 0) => (value == null ? '—' : `${fmtNum(value, digits, true)}%`);

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
