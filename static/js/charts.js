/**
 * Chart.js helpers. Colours always come from the theme tokens, so charts stay legible in light
 * and dark mode; every chart replaces the previous one drawn on the same canvas.
 *
 * Motion language: line charts are revealed by a left-to-right scanner sweep, the trend lines
 * glow, a crosshair follows the pointer, and bars grow in a staggered cascade.
 */
import { chartColors, isLight, prefersReducedMotion, token } from './theme.js';

const charts = new Map();
const SWEEP_MS = 1500;
const easeOutQuart = (x) => 1 - (1 - x) ** 4;

/** "#rrggbb" / "#rgb" / "rgb(...)" -> "rgba(r,g,b,a)" so canvas gradients accept it. */
export function withAlpha(color, alpha) {
  let c = String(color).trim();
  if (c.startsWith('#')) {
    if (c.length === 4) c = `#${[...c.slice(1)].map((h) => h + h).join('')}`;
    const n = parseInt(c.slice(1, 7), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
  }
  const m = c.match(/[\d.]+/g);
  return m ? `rgba(${m[0]},${m[1]},${m[2]},${alpha})` : c;
}

/** Scriptable vertical fade under a line: pass as `backgroundColor` with `fill: 'origin'`. */
export function areaGradient(color, top = 0.28) {
  return (context) => {
    const { chart } = context;
    const area = chart.chartArea;
    if (!area) return 'transparent';
    const g = chart.ctx.createLinearGradient(0, area.top, 0, area.bottom);
    g.addColorStop(0, withAlpha(color, top));
    g.addColorStop(1, withAlpha(color, 0));
    return g;
  };
}

/** Bars painted with a vertical gradient from the series colour to a faded base. */
export function barGradient(color, from = 1, to = 0.35) {
  return (context) => {
    const { chart } = context;
    const area = chart.chartArea;
    if (!area) return color;
    const horizontal = chart.options.indexAxis === 'y';
    const g = horizontal
      ? chart.ctx.createLinearGradient(area.left, 0, area.right, 0)
      : chart.ctx.createLinearGradient(0, area.top, 0, area.bottom);
    g.addColorStop(0, withAlpha(color, horizontal ? to : from));
    g.addColorStop(1, withAlpha(color, horizontal ? from : to));
    return g;
  };
}

// ---------------------------------------------------------------------------- Plugins
const sweepPlugin = {
  id: 'tlSweep',
  beforeDatasetsDraw(chart, _args, opts) {
    if (!opts?.enabled) return;
    const s = (chart.$sweep ||= { t0: performance.now(), done: false });
    const k = Math.min(1, (performance.now() - s.t0) / SWEEP_MS);
    s.progress = easeOutQuart(k);
    if (k >= 1) { s.done = true; return; }
    const { ctx, chartArea: a } = chart;
    ctx.save();
    ctx.beginPath();
    ctx.rect(a.left - 10, a.top - 10, (a.width + 20) * s.progress, a.height + 20);
    ctx.clip();
    s.clipped = true;
    if (!s.raf) {
      s.raf = requestAnimationFrame(() => {
        s.raf = 0;
        if (chart.ctx && !chart.$destroyed) chart.draw();
      });
    }
  },
  afterDatasetsDraw(chart, _args, opts) {
    const s = chart.$sweep;
    if (!opts?.enabled || !s?.clipped) return;
    const { ctx, chartArea: a } = chart;
    ctx.restore();
    s.clipped = false;
    // Scanner head: a thin bright line with a soft trail
    const x = a.left + a.width * s.progress;
    const accent = token('--accent');
    const trail = ctx.createLinearGradient(x - 80, 0, x, 0);
    trail.addColorStop(0, withAlpha(accent, 0));
    trail.addColorStop(1, withAlpha(accent, 0.18));
    ctx.save();
    ctx.fillStyle = trail;
    ctx.fillRect(Math.max(a.left, x - 80), a.top, Math.min(80, x - a.left), a.height);
    ctx.strokeStyle = withAlpha(accent, 0.85);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, a.top);
    ctx.lineTo(x, a.bottom);
    ctx.stroke();
    ctx.restore();
  },
  afterDestroy(chart) { chart.$destroyed = true; },
};

const glowPlugin = {
  id: 'tlGlow',
  beforeDatasetDraw(chart, args) {
    const ds = chart.data.datasets[args.index];
    if (!ds || ds.glow === false || (ds.borderWidth || 0) < 2 || chart.config.type !== 'line' || isLight()) return;
    const { ctx } = chart;
    ctx.save();
    ctx.shadowColor = withAlpha(ds.borderColor, 0.7);
    ctx.shadowBlur = 12;
    chart.$glowing = true;
  },
  afterDatasetDraw(chart) {
    if (chart.$glowing) { chart.ctx.restore(); chart.$glowing = false; }
  },
};

const crosshairPlugin = {
  id: 'tlCrosshair',
  afterDatasetsDraw(chart, _args, opts) {
    if (opts?.enabled === false || chart.config.type !== 'line') return;
    const active = chart.tooltip?.getActiveElements?.();
    if (!active?.length) return;
    const { ctx, chartArea: a } = chart;
    const x = active[0].element.x;
    const line = ctx.createLinearGradient(0, a.top, 0, a.bottom);
    const accent = token('--accent');
    line.addColorStop(0, withAlpha(accent, 0));
    line.addColorStop(0.5, withAlpha(accent, 0.65));
    line.addColorStop(1, withAlpha(accent, 0));
    ctx.save();
    ctx.strokeStyle = line;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(x, a.top);
    ctx.lineTo(x, a.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

let registered = false;
function register() {
  if (registered || !window.Chart) return;
  window.Chart.register(sweepPlugin, glowPlugin, crosshairPlugin);
  registered = true;
}

// ---------------------------------------------------------------------------- Rendering
export function renderChart(canvasId, config, ariaLabel) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  register();
  charts.get(canvasId)?.destroy();

  const options = (config.options ||= {});
  const motion = !prefersReducedMotion();
  const cartesianLine = config.type === 'line' || config.type === 'scatter';
  if (!motion) {
    options.animation = false;
  } else if (cartesianLine) {
    // The scanner sweep owns the entrance; native point animation would fight it.
    options.animation = false;
    (options.plugins ||= {}).tlSweep = { enabled: true };
  } else if (config.type === 'bar') {
    options.animation = {
      duration: 900,
      easing: 'easeOutQuart',
      delay: (ctx) => (ctx.type === 'data' && ctx.mode === 'default' ? ctx.dataIndex * 38 : 0),
    };
  }

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
    padding: 12,
    cornerRadius: 12,
    boxPadding: 5,
    usePointStyle: true,
    caretSize: 0,
    titleFont: { family: token('--font-mono'), size: 11, weight: '400' },
    bodyFont: { family: token('--font-sans'), size: 12.5 },
    ...extra,
  };
}

export function axis({ title, grid = true, ...rest } = {}) {
  const c = chartColors();
  const mono = { family: token('--font-mono'), size: 10.5 };
  return {
    grid: grid ? { color: c.grid, drawTicks: false } : { display: false },
    border: { display: false },
    ticks: { color: c.text, font: mono, padding: 8, ...(rest.ticks || {}) },
    title: title ? { display: true, text: title, color: c.text, font: { family: token('--font-sans'), size: 11.5, weight: '500' } } : { display: false },
    ...Object.fromEntries(Object.entries(rest).filter(([k]) => k !== 'ticks')),
  };
}

export function baseOptions({ legend = false, tooltip = {}, ...extra } = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: legend
        ? { display: true, labels: { color: chartColors().text, usePointStyle: true, pointStyle: 'line', boxWidth: 22, font: { family: token('--font-mono'), size: 11 } } }
        : { display: false },
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
