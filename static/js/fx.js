/**
 * Visual effects layer: live star canvas, cursor spotlight, KPI count-up and the sliding
 * navigation indicator. Everything is progressive enhancement: with reduced motion the sky is
 * drawn once and numbers appear without tweening.
 */
const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const easeOutExpo = (x) => (x === 1 ? 1 : 1 - 2 ** (-10 * x));

// ---------------------------------------------------------------------------- Star canvas
function initStarfield() {
  const canvas = document.getElementById('fxStars');
  const ctx = canvas?.getContext('2d');
  if (!ctx) return;
  let w = 0, h = 0, dpr = 1, stars = [], comet = null, raf = 0, nextComet = performance.now() + 5000;

  const starColor = () => getComputedStyle(document.documentElement).getPropertyValue('--star-rgb').trim() || '205 225 255';
  let rgb = starColor();

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const count = Math.min(220, Math.floor((w * h) / 8500));
    stars = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() ** 3 * 1.3 + 0.25,
      a: Math.random() * 0.6 + 0.25,
      tw: Math.random() * 0.002 + 0.0006,
      ph: Math.random() * 6.28,
      vx: (Math.random() * 0.5 + 0.1) * 0.012,
    }));
    draw(performance.now());
  }

  function draw(now) {
    ctx.clearRect(0, 0, w, h);
    for (const s of stars) {
      if (!reduced()) {
        s.x += s.vx;
        if (s.x > w + 2) s.x = -2;
      }
      const alpha = s.a * (0.65 + 0.35 * Math.sin(now * s.tw + s.ph));
      ctx.fillStyle = `rgb(${rgb} / ${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, 6.2832);
      ctx.fill();
    }
    if (comet) {
      const k = (now - comet.t0) / comet.dur;
      if (k >= 1) comet = null;
      else {
        const x = comet.x0 + comet.dx * k;
        const y = comet.y0 + comet.dy * k;
        const tail = ctx.createLinearGradient(x, y, x - comet.dx * 0.12, y - comet.dy * 0.12);
        tail.addColorStop(0, `rgb(${rgb} / ${(0.9 * (1 - k)).toFixed(3)})`);
        tail.addColorStop(1, `rgb(${rgb} / 0)`);
        ctx.strokeStyle = tail;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x - comet.dx * 0.12, y - comet.dy * 0.12);
        ctx.stroke();
      }
    } else if (now > nextComet && !reduced()) {
      const x0 = Math.random() * w * 0.7;
      comet = { x0, y0: Math.random() * h * 0.35, dx: w * 0.35, dy: h * 0.22, t0: now, dur: 1300 };
      nextComet = now + 8000 + Math.random() * 9000;
    }
  }

  function loop(now) {
    if (!document.hidden && !reduced()) draw(now);
    raf = requestAnimationFrame(loop);
  }

  resize();
  window.addEventListener('resize', resize);
  new MutationObserver(() => { rgb = starColor(); draw(performance.now()); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  if (!reduced()) raf = requestAnimationFrame(loop);
  window.addEventListener('pagehide', () => cancelAnimationFrame(raf));
}

// ---------------------------------------------------------------------------- Spotlight
function initSpotlight() {
  if (!window.matchMedia('(hover: hover)').matches) return;
  let frame = 0, last;
  document.addEventListener('pointermove', (event) => {
    last = event;
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      const el = last.target.closest?.('.panel, .kpi-card, .answer-card, .command-bar');
      if (!el) return;
      const r = el.getBoundingClientRect();
      el.style.setProperty('--mx', `${last.clientX - r.left}px`);
      el.style.setProperty('--my', `${last.clientY - r.top}px`);
    });
  }, { passive: true });
}

// ---------------------------------------------------------------------------- KPI count-up
const NUMBER = /^([+\-−]?)(\d[\d,]*)(?:\.(\d+))?$/;

function parse(text) {
  const m = NUMBER.exec(text.trim());
  if (!m) return null;
  const decimals = m[3]?.length ?? 0;
  const value = Number(`${m[1] === '−' ? '-' : m[1]}${m[2].replace(/,/g, '')}${m[3] ? `.${m[3]}` : ''}`);
  return { value, decimals, signed: m[1] === '+' || m[1] === '−' || m[1] === '-' };
}

function format({ value, decimals, signed }) {
  const body = Math.abs(value).toFixed(decimals);
  if (value < 0) return `−${body}`;
  return signed && value > 0 ? `+${body}` : body;
}

function initCountUp() {
  const tween = new WeakMap();
  const handle = (el) => {
    if (!el.isConnected || el.textContent === el.__fxLast) return;
    const next = parse(el.textContent);
    cancelAnimationFrame(tween.get(el));
    if (!next || reduced()) { el.__fxLast = el.textContent; return; }
    const prev = el.__fxFrom ?? { ...next, value: 0 };
    const from = prev.value;
    const final = el.textContent;
    const t0 = performance.now();
    const dur = 1100;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur);
      const shown = k === 1 ? final : format({ ...next, value: from + (next.value - from) * easeOutExpo(k) });
      el.__fxLast = shown;
      el.textContent = shown;
      if (k < 1) tween.set(el, requestAnimationFrame(step));
      else el.__fxFrom = next;
    };
    tween.set(el, requestAnimationFrame(step));
  };
  const observer = new MutationObserver((records) => {
    new Set(records.map((r) => (r.target.nodeType === 3 ? r.target.parentElement : r.target))).forEach((el) => {
      if (el?.matches?.('.kpi-value, .kpi-value-hero')) handle(el);
    });
  });
  document.querySelectorAll('.kpi-value, .kpi-value-hero').forEach((el) => observer.observe(el, { childList: true, characterData: true, subtree: true }));
  // KPI cards rendered later (dipole summary strip) are not counted: they are static HTML.
}

// ---------------------------------------------------------------------------- Nav indicator
function initNavIndicator() {
  const list = document.querySelector('.nav-list[role="tablist"]');
  if (!list) return;
  document.documentElement.classList.add('js-fx');
  const bar = document.createElement('div');
  bar.className = 'nav-indicator';
  bar.setAttribute('aria-hidden', 'true');
  list.prepend(bar);

  const place = () => {
    const active = list.querySelector('.nav-btn.active');
    if (!active || !active.offsetHeight) { bar.style.opacity = '0'; return; }
    bar.style.height = `${active.offsetHeight}px`;
    bar.style.transform = `translateY(${active.offsetTop}px)`;
    bar.style.opacity = '1';
  };
  new MutationObserver(place).observe(list, { subtree: true, attributes: true, attributeFilter: ['class'] });
  window.addEventListener('resize', place);
  document.fonts?.ready.then(place);
  requestAnimationFrame(place);
  setTimeout(place, 400);
}

// ---------------------------------------------------------------------------- Answer card scan line
function initAnswerScan() {
  const card = document.getElementById('answerCard');
  if (card && !card.querySelector('.answer-scan')) {
    const line = document.createElement('span');
    line.className = 'answer-scan';
    line.setAttribute('aria-hidden', 'true');
    card.prepend(line);
  }
}

function init() {
  initStarfield();
  initSpotlight();
  initCountUp();
  initNavIndicator();
  initAnswerScan();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
