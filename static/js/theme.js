/**
 * Light/dark theme. The first paint is set by an inline script in index.html; this module keeps
 * the choice, exposes the chart colours defined as CSS tokens and notifies listeners on change.
 */
import { storage } from './storage.js';

const STORAGE_KEY = 'terralens.theme';
const listeners = new Set();

export const isLight = () => document.documentElement.classList.contains('light');

export function setTheme(light) {
  document.documentElement.classList.toggle('light', light);
  storage.set(STORAGE_KEY, light ? 'light' : 'dark');
  applyChartDefaults();
  listeners.forEach((fn) => fn(light));
}

export const onThemeChange = (fn) => listeners.add(fn);

export const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Reads a CSS custom property from the root element. */
export const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function chartColors() {
  return {
    text: token('--chart-text'),
    grid: token('--chart-grid'),
    tooltipBg: token('--chart-tooltip-bg'),
    tooltipText: token('--chart-tooltip-text'),
    anomaly: token('--series-anomaly'),
    trend: token('--series-trend'),
    ols: token('--series-ols'),
    step: token('--series-step'),
    alert: token('--series-alert'),
    muted: token('--series-muted'),
    oni: token('--series-oni'),
    border: token('--border-strong'),
  };
}

export function applyChartDefaults() {
  if (!window.Chart) return;
  const c = chartColors();
  window.Chart.defaults.locale = 'en-US'; // decimal point, like every other number in the interface
  window.Chart.defaults.color = c.text;
  window.Chart.defaults.borderColor = c.grid;
  window.Chart.defaults.font.family = token('--font-sans');
  window.Chart.defaults.font.size = 12;
  window.Chart.defaults.animation = prefersReducedMotion() ? false : { duration: 900, easing: 'easeOutQuart' };
  window.Chart.defaults.elements.point.hoverRadius = 5;
  window.Chart.defaults.elements.line.borderCapStyle = 'round';
  window.Chart.defaults.elements.line.borderJoinStyle = 'round';
}
