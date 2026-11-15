/**
 * TerraLens observatory entry point: loads the catalog, keeps controls, URL and views in sync,
 * and drives the guided-case walkthroughs.
 */
import { getJson } from './api.js';
import { exportSeriesCsv } from './export.js';
import { applyStaticTranslations, getLang, initialLanguage, loadLocales, setLang, t } from './i18n.js';
import { regionName, units, variableShort, zoneLabel } from './labels.js';
import { readUrlState, setExpert, state, writeUrlState } from './state.js';
import { applyChartDefaults, isLight, onThemeChange, setTheme } from './theme.js';
import { clearError, fillSelect, highlight, initAlert, initDrawer, initTabs, setSwitch, showError, showView, withLoading } from './ui.js';
import { renderAcker } from './views/acker.js';
import { renderCaseList, renderCasePanel } from './views/cases.js';
import { loadDipole } from './views/dipole.js';
import { renderMl } from './views/ml.js';
import { invalidateMapSize, loadMultiscale, refreshMapTheme, renderMeteomatics, runProbe, setProbeIdle, zoomMap } from './views/multiscale.js';
import { renderSummary } from './views/overview.js';
import { renderSidebarBenchmark, renderTelemetry } from './views/telemetry.js';
import { renderTrends } from './views/trends.js';

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------- Controls
function regionOptions() {
  return Object.keys(state.catalog.regions).map((id) => ({ value: id, label: regionName(id) }));
}

function populateRegionSelect(select, value) {
  const groups = {};
  Object.entries(state.catalog.regions).forEach(([id, region]) => {
    (groups[region.zone] ||= []).push(new Option(regionName(id), id));
  });
  select.replaceChildren(
    ...Object.entries(groups).map(([zone, options]) => {
      const group = document.createElement('optgroup');
      group.label = zoneLabel(zone);
      group.append(...options);
      return group;
    }),
  );
  select.value = value;
}

function yearOptions(from, to) {
  return Array.from({ length: to - from + 1 }, (_, i) => ({ value: from + i, label: String(from + i) }));
}

function populateControls() {
  const [first, last] = state.catalog.data_period;
  const span = state.catalog.min_window_years - 1;
  const variables = Object.keys(state.catalog.variables).map((code) => ({ value: code, label: `${variableShort(code)} (${units(code)})` }));

  populateRegionSelect($('regionSelect'), state.region);
  fillSelect($('variableSelect'), variables, state.variable);
  fillSelect($('startYearInput'), yearOptions(first, last - span), state.startYear);
  fillSelect($('endYearInput'), yearOptions(first + span, last), state.endYear);
  fillSelect($('dipoleRegionA'), regionOptions(), state.dipoleA);
  fillSelect($('dipoleRegionB'), regionOptions(), state.dipoleB);
  fillSelect($('dipoleVariable'), variables, state.dipoleVariable);
  fillSelect($('gridVarSelect'), state.catalog.map_variables.map((code) => ({ value: code, label: variableShort(code) })), state.gridVariable);
  $('mmParamSelect').value = state.mmParam;
}

function syncControls() {
  $('regionSelect').value = state.region;
  $('variableSelect').value = state.variable;
  $('startYearInput').value = String(state.startYear);
  $('endYearInput').value = String(state.endYear);
  $('dipoleRegionA').value = state.dipoleA;
  $('dipoleRegionB').value = state.dipoleB;
  $('dipoleVariable').value = state.dipoleVariable;
  $('gridVarSelect').value = state.gridVariable;

  setSwitch($('btnCeresGuard'), state.homogeneous);
  setSwitch($('btnExcludeExtremes'), state.excludeExtremes);
  $('subCeresGuard').textContent = t(state.homogeneous ? 'ctrl.ceres.on' : 'ctrl.ceres.off');
  $('subExcludeExtremes').textContent = state.excludeExtremes
    ? t('ctrl.extremes.on', { n: state.dashboard?.filters.n_excluded_months ?? '…' })
    : t('ctrl.extremes.off');
  $('btnExpert').setAttribute('aria-pressed', String(state.expert));
  $('btnExpert').classList.toggle('active', state.expert);
  $('btnProbeMode').setAttribute('aria-pressed', String(state.probeMode));
  $('btnProbeMode').classList.toggle('active', state.probeMode);
  $('langLabel').textContent = getLang() === 'en' ? 'ES' : 'EN';
  $('themeLabel').textContent = t(isLight() ? 'ctrl.theme.dark' : 'ctrl.theme.light');
}

const persistUrl = () => writeUrlState(getLang());

/** Keeps the year window valid: start before end and at least min_window_years long. */
function clampWindow(changed) {
  const span = state.catalog.min_window_years - 1;
  const [first, last] = state.catalog.data_period;
  if (state.endYear - state.startYear < span) {
    if (changed === 'start') state.endYear = Math.min(last, state.startYear + span);
    else state.startYear = Math.max(first, state.endYear - span);
  }
}

// ---------------------------------------------------------------------------- Data loading
const mapHandlers = {
  onSelectRegion: (regionId) => changeSeries({ region: regionId }),
  onProbe: () => runProbe(),
};

async function refreshDashboard() {
  const data = await withLoading(
    () =>
      getJson(
        `/api/full-dashboard/${state.region}`,
        {
          variable: state.variable,
          start_year: state.startYear,
          end_year: state.endYear,
          exclude_ml_extremes: state.excludeExtremes,
          homogeneous_ceres: state.homogeneous,
        },
        { key: 'dashboard' },
      ),
    { onError: (err) => showError(err, refreshDashboard) },
  );
  if (!data) return;
  clearError();
  state.dashboard = data;
  renderDashboardViews();
  syncControls();
}

function renderDashboardViews() {
  const d = state.dashboard;
  if (!d) return;
  renderSummary(d);
  renderTrends(d, { onSelectVariable: (variable) => changeSeries({ variable }) });
  renderMl(d);
  renderAcker(d);
}

const VIEW_LOADERS = {
  'tab-dipole': () => loadDipole(),
  'tab-multiscale': () => loadMultiscale(mapHandlers),
};

async function loadActiveView() {
  if (state.view === 'tab-telemetry') renderTelemetry(state.manifest);
  const loader = VIEW_LOADERS[state.view];
  if (!loader) return;
  const data = await withLoading(loader, { onError: (err) => showError(err, loadActiveView) });
  if (data) clearError();
}

async function selectView(view) {
  state.view = view;
  showView(view);
  persistUrl();
  await loadActiveView();
  if (view === 'tab-multiscale') invalidateMapSize();
}

/** Changes the analysed series; a new region or variable ends the guided case. */
async function changeSeries(changes) {
  const leavesCase = ('region' in changes && changes.region !== state.region) || ('variable' in changes && changes.variable !== state.variable);
  Object.assign(state, changes);
  if (leavesCase) closeCase();
  syncControls();
  persistUrl();
  await refreshDashboard();
  if (state.view === 'tab-multiscale') await loadActiveView();
}

// ---------------------------------------------------------------------------- Guided cases
const caseById = (id) => state.catalog.cases.find((c) => c.id === id);

function renderCases() {
  renderCaseList(state.catalog.cases, state.caseId, activateCase);
  renderCasePanel(caseById(state.caseId), state.caseStep);
}

async function goToStep(index) {
  const current = caseById(state.caseId);
  if (!current) return;
  state.caseStep = Math.max(0, Math.min(index, current.steps.length - 1));
  renderCasePanel(current, state.caseStep);
  const step = current.steps[state.caseStep];
  if (step.view !== state.view) await selectView(step.view);
  persistUrl();
  highlight(step.target);
}

function applyCaseSetup(setup) {
  Object.assign(state, {
    region: setup.region,
    variable: setup.variable,
    startYear: setup.start_year,
    endYear: state.catalog.data_period[1],
    homogeneous: setup.homogeneous_splice,
    excludeExtremes: false,
  });
  if (setup.dipole) [state.dipoleA, state.dipoleB, state.dipoleVariable] = setup.dipole;
}

async function activateCase(caseId) {
  Object.assign(state, { caseId, caseStep: 0 });
  applyCaseSetup(caseById(caseId).setup);
  syncControls();
  renderCases();
  await refreshDashboard();
  await goToStep(0);
}

function closeCase() {
  state.caseId = null;
  state.caseStep = 0;
  renderCases();
  persistUrl();
}

// ---------------------------------------------------------------------------- Language and theme
async function rerenderAll() {
  applyStaticTranslations();
  document.title = t('app.title');
  populateControls();
  syncControls();
  renderCases();
  renderSidebarBenchmark(state.manifest);
  showView(state.view);
  renderDashboardViews();
  setProbeIdle();
  await loadActiveView();
}

// ---------------------------------------------------------------------------- Events
function on(id, event, handler) {
  $(id).addEventListener(event, handler);
}

function wireEvents() {
  on('regionSelect', 'change', (e) => changeSeries({ region: e.target.value }));
  on('variableSelect', 'change', (e) => changeSeries({ variable: e.target.value }));
  on('startYearInput', 'change', (e) => {
    state.startYear = Number(e.target.value);
    clampWindow('start');
    changeSeries({});
  });
  on('endYearInput', 'change', (e) => {
    state.endYear = Number(e.target.value);
    clampWindow('end');
    changeSeries({});
  });
  on('btnCeresGuard', 'click', () => changeSeries({ homogeneous: !state.homogeneous }));
  on('btnExcludeExtremes', 'click', () => changeSeries({ excludeExtremes: !state.excludeExtremes }));
  on('btnExpert', 'click', () => {
    setExpert(!state.expert);
    syncControls();
  });
  on('btnLang', 'click', () => {
    setLang(getLang() === 'en' ? 'es' : 'en');
    persistUrl();
    rerenderAll();
  });
  on('btnTheme', 'click', () => setTheme(!isLight()));
  on('btnExportCsv', 'click', () => exportSeriesCsv(state.dashboard));

  const dipoleChange = (field) => (e) => {
    state[field] = e.target.value;
    persistUrl();
    if (state.dipoleA === state.dipoleB) {
      showError({ status: 422, message: t('dipole.sameRegion') });
      return;
    }
    loadActiveView();
  };
  on('dipoleRegionA', 'change', dipoleChange('dipoleA'));
  on('dipoleRegionB', 'change', dipoleChange('dipoleB'));
  on('dipoleVariable', 'change', dipoleChange('dipoleVariable'));

  on('gridVarSelect', 'change', (e) => {
    state.gridVariable = e.target.value;
    persistUrl();
    loadActiveView();
  });
  on('mmParamSelect', 'change', (e) => {
    state.mmParam = e.target.value;
    renderMeteomatics();
  });
  on('btnZoomEcuador', 'click', () => zoomMap('ecuador'));
  on('btnZoomGlobal', 'click', () => zoomMap('global'));
  on('btnProbeMode', 'click', () => {
    state.probeMode = !state.probeMode;
    syncControls();
  });
  on('probeForm', 'submit', (e) => {
    e.preventDefault();
    runProbe();
  });

  on('btnCasePrev', 'click', () => goToStep(state.caseStep - 1));
  on('btnCaseNext', 'click', () => {
    const current = caseById(state.caseId);
    if (current && state.caseStep >= current.steps.length - 1) closeCase();
    else goToStep(state.caseStep + 1);
  });
  on('btnCaseClose', 'click', closeCase);

  onThemeChange(() => {
    syncControls();
    refreshMapTheme();
    renderDashboardViews();
    loadActiveView();
  });
}

// ---------------------------------------------------------------------------- Start
async function init() {
  applyChartDefaults();
  initAlert();
  initDrawer();
  try {
    await loadLocales();
    setLang(initialLanguage());
    document.title = t('app.title');
    [state.catalog, state.manifest] = await Promise.all([getJson('/api/catalog'), getJson('/api/ml-manifest')]);
  } catch (err) {
    showError(err, () => location.reload());
    return;
  }

  const params = readUrlState(state.catalog);
  // A shared link such as ?case=sensor_splice carries only the case: rebuild its setup.
  const sharedCase = caseById(state.caseId);
  if (sharedCase) {
    state.caseStep = Math.min(state.caseStep, sharedCase.steps.length - 1);
    if (!params.has('region')) applyCaseSetup(sharedCase.setup);
    if (!params.has('tab')) state.view = sharedCase.steps[state.caseStep].view;
  }
  setExpert(state.expert);
  populateControls();
  syncControls();
  wireEvents();
  initTabs(selectView);
  renderCases();
  renderSidebarBenchmark(state.manifest);
  showView(state.view);
  persistUrl();

  await refreshDashboard();
  await loadActiveView();
  const current = caseById(state.caseId);
  if (current) highlight(current.steps[state.caseStep]?.target);
}

init();
