/** Map and vegetation view: grid trend map, MODIS NDVI, Meteomatics snapshots, STAC provenance, live probe. */
import { getJson } from '../api.js';
import { axis, baseOptions, divergingColor, paletteStops, renderChart } from '../charts.js';
import { escapeHtml, fmtNum, pText, qText } from '../format.js';
import { t } from '../i18n.js';
import { decadeUnits, regionName, variableShort, verdictLabel, verdictTone, zoneLabel } from '../labels.js';
import { state } from '../state.js';
import { chartColors, isLight, prefersReducedMotion, token } from '../theme.js';

const $ = (id) => document.getElementById(id);
const ECUADOR_VIEW = { center: [-1.75, -78.25], zoom: 6 };
const GLOBAL_VIEW = { center: [15, -75], zoom: 2 };
const CELL_HALF_LAT = 0.25;
const CELL_HALF_LON = 0.3125;
const TILE_ROOT = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas';
const TILE_ATTRIBUTION = 'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors | Data: NASA POWER';

let map = null;
let baseLayer = null;
let referenceLayer = null;
let gridLayer = null;
let lastData = null;
let handlers = { onSelectRegion: () => {}, onProbe: () => {} };

function tileUrls() {
  const tone = isLight() ? 'Light' : 'Dark';
  return {
    base: `${TILE_ROOT}/World_${tone}_Gray_Base/MapServer/tile/{z}/{y}/{x}`,
    reference: `${TILE_ROOT}/World_${tone}_Gray_Reference/MapServer/tile/{z}/{y}/{x}`,
  };
}

function ensureMap(viewHandlers) {
  handlers = viewHandlers;
  if (map || !window.L) return;
  map = window.L.map('ecuadorMap', { ...ECUADOR_VIEW, preferCanvas: true });
  const urls = tileUrls();
  baseLayer = window.L.tileLayer(urls.base, { attribution: TILE_ATTRIBUTION, maxZoom: 13 }).addTo(map);
  referenceLayer = window.L.tileLayer(urls.reference, { maxZoom: 13, opacity: 0.65 }).addTo(map);
  gridLayer = window.L.layerGroup().addTo(map);
  map.on('click', (event) => {
    if (!state.probeMode) return;
    $('probeLat').value = event.latlng.lat.toFixed(2);
    $('probeLon').value = event.latlng.lng.toFixed(2);
    handlers.onProbe();
  });
}

export function refreshMapTheme() {
  if (!map) return;
  const urls = tileUrls();
  baseLayer.setUrl(urls.base);
  referenceLayer.setUrl(urls.reference);
}

export function zoomMap(scope) {
  if (!map) return;
  const view = scope === 'global' ? GLOBAL_VIEW : ECUADOR_VIEW;
  if (prefersReducedMotion()) map.setView(view.center, view.zoom);
  else map.flyTo(view.center, view.zoom, { duration: 0.8 });
  $('btnZoomEcuador').classList.toggle('active', scope !== 'global');
  $('btnZoomGlobal').classList.toggle('active', scope === 'global');
}

export const invalidateMapSize = () => map?.invalidateSize();

function renderLegend(variable) {
  const meta = state.catalog.variables[variable].map;
  const stops = paletteStops(meta.palette);
  const limit = meta.scale_limit;
  $('mapLegend').innerHTML = `
    <div class="map-legend-scale">
      <span>${escapeHtml(t('map.legendTitle', { units: decadeUnits(variable) }))}</span>
      <div class="map-legend-bar" style="background:linear-gradient(90deg, ${stops.join(', ')});"></div>
      <div class="map-legend-ticks"><span>${fmtNum(-limit, 2)}</span><span>0</span><span>+${fmtNum(limit, 2)}</span></div>
    </div>
    <span><span class="legend-cell" style="background:${stops[3]};outline:2px solid var(--text-primary);"></span>${escapeHtml(t('map.legendSignificant'))}</span>
    <span><span class="legend-cell" style="background:${stops[3]};opacity:0.35;"></span>${escapeHtml(t('map.legendNoise'))}</span>
    <span><span class="legend-cell" style="border-radius:50%;width:10px;background:var(--accent);"></span>${escapeHtml(t('map.legendPin'))}</span>`;
}

function renderGrid(cells, variable) {
  gridLayer.clearLayers();
  const { scale_limit: limit, palette } = state.catalog.variables[variable].map;
  const outline = token('--text-primary');
  cells.forEach((cell) => {
    const color = divergingColor(cell.theil_sen_decade, limit, palette);
    const rect = window.L.rectangle(
      [[cell.lat - CELL_HALF_LAT, cell.lon - CELL_HALF_LON], [cell.lat + CELL_HALF_LAT, cell.lon + CELL_HALF_LON]],
      {
        color: cell.significant ? outline : color,
        weight: cell.significant ? 1.2 : 0.4,
        fillColor: color,
        fillOpacity: cell.significant ? 0.75 : 0.3,
      },
    );
    rect.bindTooltip(
      `<strong>${cell.lat.toFixed(2)}, ${cell.lon.toFixed(2)}</strong> · ${fmtNum(cell.elevation_m, 0)} m<br/>` +
        `${escapeHtml(variableShort(variable))}: <strong>${fmtNum(cell.theil_sen_decade, 3, true)}</strong> ${escapeHtml(decadeUnits(variable))}<br/>` +
        `${pText(cell.p_hamed_rao)} · ${qText(cell.q_fdr)} · ${escapeHtml(t(cell.significant ? 'scan.significant' : 'scan.notSignificant'))}`,
    );
    gridLayer.addLayer(rect);
  });

  Object.entries(state.catalog.regions).forEach(([id, region]) => {
    const selected = id === state.region;
    const marker = window.L.circleMarker([region.lat, region.lon], {
      radius: selected ? 8 : 6, color: '#ffffff', weight: 2,
      fillColor: selected ? token('--status-green') : token('--accent'), fillOpacity: 0.95,
    });
    marker.bindTooltip(`<strong>${escapeHtml(regionName(id))}</strong><br/>${escapeHtml(zoneLabel(region.zone))}<br/><em>${escapeHtml(t('map.pinHint'))}</em>`);
    marker.on('click', (event) => {
      window.L.DomEvent.stopPropagation(event);
      handlers.onSelectRegion(id);
    });
    gridLayer.addLayer(marker);
  });
  renderLegend(variable);
}

function renderModis(modis) {
  $('modisTitle').textContent = t('modis.title', { region: regionName(modis.region_id), n: modis.records.length });
  $('modisNotice').hidden = !modis.is_fallback;
  if (modis.is_fallback) $('modisNotice').textContent = t('modis.fallback', { region: regionName(state.region), fallback: regionName(modis.region_id) });
  const c = chartColors();
  renderChart(
    'modisChart',
    {
      type: 'line',
      data: {
        labels: modis.records.map((r) => r.calendar_date),
        datasets: [{ label: 'NDVI', data: modis.records.map((r) => r.ndvi), borderColor: c.trend, borderWidth: 1.5, pointRadius: 1.5, tension: 0 }],
      },
      options: baseOptions({ scales: { x: axis({ grid: false, ticks: { maxTicksLimit: 7 } }), y: axis({ title: 'NDVI' }) } }),
    },
    t('aria.modis', { region: regionName(modis.region_id) }),
  );
}

export function renderMeteomatics(series = lastData?.meteomatics_1km) {
  if (!series) return;
  const c = chartColors();
  const param = state.mmParam;
  const colors = [c.anomaly, c.alert, c.ols, c.trend];
  const regions = Object.keys(series);
  const labels = series[regions[0]]?.map((r) => r.date.slice(0, 10)) ?? [];
  renderChart(
    'meteomaticsChart',
    {
      type: 'line',
      data: {
        labels,
        datasets: regions.map((id, i) => {
          const selected = id === state.region;
          return {
            label: regionName(id),
            data: series[id].map((r) => r[param]),
            borderColor: selected || !(state.region in series) ? colors[i % colors.length] : `${colors[i % colors.length]}55`,
            borderWidth: selected ? 2.4 : 1.3,
            pointRadius: selected ? 2 : 0,
            tension: 0,
          };
        }),
      },
      options: baseOptions({ legend: true, scales: { x: axis({ grid: false, ticks: { maxTicksLimit: 8 } }), y: axis() } }),
    },
    t('aria.meteomatics'),
  );
}

function renderStac(items) {
  const tbody = $('stacTableBody');
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="4" class="empty-state">${escapeHtml(t('stac.none'))}</td></tr>`;
    return;
  }
  tbody.replaceChildren(
    ...items.map((item) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td class="tabular" style="font-size:11px;">${escapeHtml(item.stac_id)}</td><td class="tabular">${escapeHtml(item.datetime)}</td><td class="tabular">${escapeHtml(item.tile_grid)}</td><td>${escapeHtml(item.platform)}</td>`;
      return tr;
    }),
  );
}

export async function loadMultiscale(viewHandlers) {
  ensureMap(viewHandlers);
  const data = await getJson('/api/multiscale', { region_id: state.region, grid_variable: state.gridVariable }, { key: 'multiscale' });
  lastData = data;
  if (map) renderGrid(data.grid_cells, data.grid_variable);
  renderModis(data.modis);
  renderMeteomatics(data.meteomatics_1km);
  renderStac(data.planetary_computer_stac);
  setTimeout(invalidateMapSize, 120);
  return data;
}

export function setProbeIdle() {
  $('liveProbeResult').textContent = t('probe.idle');
}

export async function runProbe() {
  const lat = Number($('probeLat').value);
  const lon = Number($('probeLon').value);
  const box = $('liveProbeResult');
  box.textContent = t('probe.loading', { lat: fmtNum(lat, 2), lon: fmtNum(lon, 2) });
  try {
    const d = await getJson('/api/live-probe', { lat, lon, variable: state.variable }, { key: 'probe' });
    const tone = verdictTone(d.verdict.code);
    const top = d.top_anomalies[0];
    box.innerHTML = `
      <div style="color:var(--text-primary);font-weight:600;margin-bottom:3px;">${fmtNum(d.meta.lat, 2)}, ${fmtNum(d.meta.lon, 2)} · ${fmtNum(d.meta.elevation_m, 0)} m · ${escapeHtml(variableShort(d.variable))}</div>
      <div class="tabular">${fmtNum(d.monthly_mk.theil_sen_decade, 3, true)} ${escapeHtml(decadeUnits(d.variable))} · ${pText(d.monthly_mk.p_hamed_rao)} · <span class="tone-${tone}">${escapeHtml(verdictLabel(d.verdict.code))}</span></div>
      <div class="tabular">${escapeHtml(t('probe.extra', { date: d.pettitt.change_date, top: top ? top.date : '—' }))}</div>`;
  } catch (err) {
    if (err?.name === 'AbortError') return;
    box.textContent = t('probe.error', { message: err.message });
  }
}
