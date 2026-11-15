/** Shared interface behaviour: views (ARIA tabs), loading state, error alert, drawer, highlights. */
import { t } from './i18n.js';
import { state, SUMMARY_VIEWS } from './state.js';
import { prefersReducedMotion } from './theme.js';

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------- Errors
let retryHandler = null;

export function initAlert() {
  $('btnAlertClose').addEventListener('click', clearError);
  $('btnAlertRetry').addEventListener('click', () => {
    const retry = retryHandler;
    clearError();
    retry?.();
  });
}

export function showError(err, retry = null) {
  const status = err?.status;
  const key = status === 0 ? 'alert.network' : status === 429 ? 'alert.rateLimited' : status >= 500 ? 'alert.server' : 'alert.request';
  const detail = err?.detail;
  $('appAlertText').textContent = detail?.code === 'short_effective_window'
    ? t('alert.effectiveWindow', detail)
    : t(key, { message: err?.message || '' });
  retryHandler = retry;
  $('btnAlertRetry').hidden = !retry;
  $('appAlert').hidden = false;
}

export function clearError() {
  $('appAlert').hidden = true;
  retryHandler = null;
}

// ---------------------------------------------------------------------------- Loading
let pending = 0;

export function setLoading(isLoading) {
  pending = Math.max(0, pending + (isLoading ? 1 : -1));
  $('loadingIndicator').style.display = pending > 0 ? 'inline-flex' : 'none';
  document.body.classList.toggle('is-loading', pending > 0);
}

/** Runs an async task with the loading indicator; aborts are silent, other errors are shown. */
export async function withLoading(task, { onError } = {}) {
  setLoading(true);
  try {
    return await task();
  } catch (err) {
    if (err?.name !== 'AbortError') onError?.(err);
    return undefined;
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------- Views (tabs)
const tabButtons = () => [...document.querySelectorAll('.nav-btn[role="tab"]')];

export function initTabs(onSelect) {
  tabButtons().forEach((btn, index, all) => {
    btn.addEventListener('click', () => onSelect(btn.dataset.tab));
    btn.addEventListener('keydown', (event) => {
      const moves = { ArrowDown: index + 1, ArrowUp: index - 1, Home: 0, End: all.length - 1 };
      if (!(event.key in moves)) return;
      event.preventDefault();
      const target = all[(moves[event.key] + all.length) % all.length];
      target.focus();
      if (!document.body.classList.contains('drawer-open')) onSelect(target.dataset.tab);
    });
  });
}

export function showView(view) {
  tabButtons().forEach((btn) => {
    const active = btn.dataset.tab === view;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
    btn.tabIndex = active ? 0 : -1;
    if (active) $('currentViewTitle').textContent = btn.querySelector('[data-i18n]').textContent;
  });
  document.querySelectorAll('.workspace-view').forEach((panel) => panel.classList.toggle('active', panel.id === view));
  document.querySelectorAll('[data-series-control]').forEach((control) => { control.hidden = view === 'tab-dipole'; });
  syncSeriesAvailability();
  $('viewGuideText').textContent = t(`viewGuide.${view}`);
  closeDrawer($('workspace'));
}

export function syncSeriesAvailability() {
  const available = Boolean(state.dashboard);
  $('summarySection').hidden = !available || !SUMMARY_VIEWS.includes(state.view);
  SUMMARY_VIEWS.forEach((view) => { $(view).hidden = !available; });
  $('btnExportCsv').disabled = !available;
}

// ---------------------------------------------------------------------------- Drawer (small screens)
export function initDrawer() {
  const mobileLayout = window.matchMedia('(max-width: 960px)');
  $('btnMenu').addEventListener('click', () => {
    setDrawer(true);
    (document.querySelector('.nav-btn.active') ?? $('btnDrawerClose')).focus();
  });
  ['btnDrawerClose', 'drawerBackdrop'].forEach((id) => {
    $(id).addEventListener('click', () => closeDrawer());
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeDrawer();
  });
  mobileLayout.addEventListener('change', (event) => {
    if (!event.matches) closeDrawer(document.querySelector('.nav-btn.active'));
  });
}

function setDrawer(open) {
  document.body.classList.toggle('drawer-open', open);
  $('btnMenu').setAttribute('aria-expanded', String(open));
  $('mainArea').inert = open;
}

export function closeDrawer(focusTarget = $('btnMenu')) {
  if (!document.body.classList.contains('drawer-open')) return;
  setDrawer(false);
  focusTarget?.focus({ preventScroll: true });
}

// ---------------------------------------------------------------------------- Helpers
export function setSwitch(button, on) {
  button.classList.toggle('active', on);
  button.setAttribute('aria-checked', String(on));
}

export function fillSelect(select, options, value) {
  select.replaceChildren(...options.map(({ value: v, label }) => new Option(label, v)));
  if (value !== undefined) select.value = String(value);
}

let highlightTimer;

export function highlight(elementId) {
  const el = $(elementId);
  if (!el) return;
  clearTimeout(highlightTimer);
  document.querySelector('.case-highlight')?.classList.remove('case-highlight');
  el.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'center' });
  void el.offsetWidth; // restart the animation
  el.classList.add('case-highlight');
  highlightTimer = setTimeout(() => el.classList.remove('case-highlight'), 2200);
}
