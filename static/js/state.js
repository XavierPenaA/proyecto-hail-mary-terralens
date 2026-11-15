/**
 * Application state and its mirror in the page address, so any view can be shared as a link.
 * Only values that differ from the defaults are written to the URL.
 */
import { storage } from './storage.js';

export const VIEWS = ['tab-trends', 'tab-ml', 'tab-acker', 'tab-dipole', 'tab-multiscale', 'tab-telemetry', 'tab-guide', 'tab-about'];
export const SUMMARY_VIEWS = ['tab-trends', 'tab-ml', 'tab-acker'];
const EXPERT_KEY = 'terralens.expert';

export const DEFAULTS = Object.freeze({
  region: 'EC_Andes_Paute',
  variable: 'T2M',
  startYear: 1981,
  endYear: 2025,
  homogeneous: true,
  excludeExtremes: false,
  view: 'tab-trends',
  caseId: null,
  caseStep: 0,
  dipoleA: 'EC_Coast_Guayas',
  dipoleB: 'BR_Amazon_Manaus',
  dipoleVariable: 'PRECTOTCORR',
  gridVariable: 'T2M',
  mmParam: 'soil_moisture_deficit:mm',
});

export const state = {
  ...DEFAULTS,
  expert: storage.get(EXPERT_KEY) === '1',
  probeMode: false,
  catalog: null,
  manifest: null,
  dashboard: null,
};

// [state field, URL parameter, type]
const URL_FIELDS = [
  ['region', 'region', 'string'],
  ['variable', 'var', 'string'],
  ['startYear', 'from', 'int'],
  ['endYear', 'to', 'int'],
  ['homogeneous', 'ceres', 'bool'],
  ['excludeExtremes', 'noext', 'bool'],
  ['view', 'tab', 'string'],
  ['caseId', 'case', 'string'],
  ['caseStep', 'step', 'int'],
  ['dipoleA', 'a', 'string'],
  ['dipoleB', 'b', 'string'],
  ['dipoleVariable', 'dvar', 'string'],
  ['gridVariable', 'layer', 'string'],
];

const parsers = {
  string: (v) => v,
  int: (v) => (/^-?\d+$/.test(v) ? Number(v) : undefined),
  bool: (v) => (v === '1' ? true : v === '0' ? false : undefined),
};
const serializers = { string: String, int: String, bool: (v) => (v ? '1' : '0') };

/** Validates a candidate state against the catalog; invalid values fall back to the defaults. */
function isValid(field, value, catalog) {
  const [first, last] = catalog.data_period;
  switch (field) {
    case 'region':
    case 'dipoleA':
    case 'dipoleB':
      return value in catalog.regions;
    case 'variable':
    case 'dipoleVariable':
      return value in catalog.variables;
    case 'gridVariable':
      return catalog.map_variables.includes(value);
    case 'startYear':
    case 'endYear':
      return value >= first && value <= last;
    case 'view':
      return VIEWS.includes(value);
    case 'caseId':
      return catalog.cases.some((c) => c.id === value);
    case 'caseStep':
      return value >= 0;
    default:
      return true;
  }
}

/** Reads the address into the state and returns the raw parameters. */
export function readUrlState(catalog) {
  const params = new URLSearchParams(location.search);
  URL_FIELDS.forEach(([field, param, type]) => {
    if (!params.has(param)) return;
    const value = parsers[type](params.get(param));
    if (value !== undefined && isValid(field, value, catalog)) state[field] = value;
  });
  if (state.endYear - state.startYear + 1 < catalog.min_window_years) {
    state.startYear = DEFAULTS.startYear;
    state.endYear = DEFAULTS.endYear;
  }
  if (params.get('expert') === '1') setExpert(true);
  return params;
}

export function writeUrlState(lang) {
  const params = new URLSearchParams();
  URL_FIELDS.forEach(([field, param, type]) => {
    const value = state[field];
    if (value !== null && value !== DEFAULTS[field]) params.set(param, serializers[type](value));
  });
  if (lang && lang !== 'en') params.set('lang', lang);
  const query = params.toString();
  history.replaceState(null, '', query ? `?${query}` : location.pathname);
}

export function setExpert(on) {
  state.expert = on;
  document.body.classList.toggle('expert', on);
  storage.set(EXPERT_KEY, on ? '1' : '0');
}
