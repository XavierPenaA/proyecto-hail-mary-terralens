/**
 * Chart.js helpers. Colours always come from the theme tokens, so charts stay legible in light
 * and dark mode; every chart replaces the previous one drawn on the same canvas.
 */
import { chartColors } from './theme.js';

const charts = new Map();

export function renderChart(canvasId, config, ariaLabel) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  charts.get(canvasId)?.destroy();
  const chart = new window.Chart(canvas.getContext('2d'), config);
  charts.set(canvasId, chart);
  if (ariaLabel) canvas.setAttribute('aria-label', ariaLabel);
  return chart;
}

export function tooltipOptions(extra = {}) {
  const c = chartColors();
  return {
    backgroundColor: c.tooltipBg,
    titleColor: c.tooltipText,
    bodyColor: c.tooltipText,
    borderColor: c.border,
    borderWidth: 1,
    ...extra,
  };
}

export function axis({ title, grid = true, ...rest } = {}) {
  const c = chartColors();
  return {
    grid: grid ? { color: c.grid } : { display: false },
    ticks: { color: c.text, ...(rest.ticks || {}) },
    title: title ? { display: true, text: title, color: c.text } : { display: false },
    ...Object.fromEntries(Object.entries(rest).filter(([k]) => k !== 'ticks')),
  };
}

export function baseOptions({ legend = false, tooltip = {}, ...extra } = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: legend ? { display: true, labels: { color: chartColors().text, boxWidth: 12 } } : { display: false },
      tooltip: tooltipOptions(tooltip),
    },
    ...extra,
  };
}

/** Diverging colour scale through white, for values in [-limit, +limit]. */
const PALETTES = {
  temperature: ['#2166ac', '#67a9cf', '#f7f7f7', '#ef8a62', '#b2182b'],
  moisture: ['#8c510a', '#d8b365', '#f5f5f5', '#5ab4ac', '#01665e'],
};

export const paletteStops = (name) => PALETTES[name] || PALETTES.temperature;

export function divergingColor(value, limit, paletteName) {
  const stops = paletteStops(paletteName);
  const x = Math.max(-1, Math.min(1, value / limit));
  const pos = ((x + 1) / 2) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(pos));
  const f = pos - i;
  const a = stops[i].match(/\w\w/g).map((h) => parseInt(h, 16));
  const b = stops[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  const rgb = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${rgb.join(',')})`;
}
