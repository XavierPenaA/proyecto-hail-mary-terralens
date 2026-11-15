/**
 * Human-readable labels for catalog entries and for the stable codes returned by the API.
 * Codes never reach the screen directly: they always go through the locale files.
 */
import { getLang, t } from './i18n.js';
import { state } from './state.js';

const isSpanish = () => getLang() === 'es';

export const regionName = (id) => {
  const region = state.catalog?.regions[id];
  if (!region) return id;
  return isSpanish() ? region.name_es : region.name;
};

export const variableMeta = (code) => state.catalog?.variables[code];
export const variableShort = (code) => {
  const v = variableMeta(code);
  return v ? (isSpanish() ? v.short_name_es : v.short_name) : code;
};
export const variableName = (code) => {
  const v = variableMeta(code);
  return v ? (isSpanish() ? v.name_es : v.name) : code;
};
export const units = (code) => {
  const v = variableMeta(code);
  return v ? (isSpanish() ? v.units_es : v.units) : '';
};
export const decadeUnits = (code) => {
  const v = variableMeta(code);
  return v ? (isSpanish() ? v.decade_units_es : v.decade_units) : '';
};

/** 'sig' | 'marginal' | 'none': drives the tone classes (always paired with text). */
export function verdictTone(code) {
  if (code === 'step_qualified') return 'marginal';
  if (code?.startsWith('sig_')) return 'sig';
  if (code?.startsWith('marginal_')) return 'marginal';
  return 'none';
}

export function qualifiedVerdict(code, warnings) {
  const hasSlope = code.startsWith('sig_') || code.startsWith('marginal_');
  return hasSlope && warnings.some((warning) => warning === 'step_dominated' || warning === 'known_splice_in_window')
    ? 'step_qualified' : code;
}

export const verdictLabel = (code) => t(`verdict.${code}`);
export const warningLabel = (code, params) => t(`warning.${code}`, params);
export const eventLabel = (code) => t(`event.${code}`);
export const oniPhaseLabel = (code) => (code ? t(`oni.${code}`) : '');
export const regimeLabel = (code) => t(`regime.${code}`);
export const sensitivityLabel = (code) => t(`sensitivity.${code}`);
export const zoneLabel = (code) => t(`zone.${code}`);
export const seasonLabel = (code) => t(`season.${code}`);
export const monthLabel = (month) => t(`month.${month}`);

/** Runtime driver codes look like SOURCE_lagK (e.g. PREC_lag1, ONI_lag3). */
export function featureLabel(code) {
  const match = /^([A-Z0-9]+)_lag(\d+)$/.exec(code);
  if (!match) return code;
  const [, source, lag] = match;
  const name = t(`feature.${source}`);
  return Number(lag) === 0 ? t('feature.now', { name }) : t('feature.lagged', { name, k: lag });
}
