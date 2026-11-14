import math
import os
import sys
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

W, H = 1280, 720
FPS = 30
TOTAL_FRAMES = 900  # Exact 30.00 seconds

# Deliberate Orbital Observatory Palette (BGR for OpenCV, RGB for PIL)
BG_RGB = (7, 11, 18)
SURFACE_RGB = (15, 23, 38)
SURFACE2_RGB = (21, 32, 52)
INK_RGB = (244, 247, 251)
MUTED_RGB = (148, 163, 184)
GRID_RGB = (22, 34, 54)
ACCENT_RGB = (56, 189, 248)     # Sky Cyan
EMERALD_RGB = (16, 185, 129)    # Signal Emerald
AMBER_RGB = (245, 158, 11)      # Breakpoint Amber
ROSE_RGB = (244, 63, 94)        # Trap / Alert Rose


def bgr(rgb: tuple) -> tuple:
    return (rgb[2], rgb[1], rgb[0])


def load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        paths = [r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf"]
    elif bold:
        paths = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"]
    else:
        paths = [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def make_canvas(phase_t: float) -> np.ndarray:
    """Clean dark orbital grid canvas with subtle moving longitude arc."""
    canvas = np.full((H, W, 3), bgr(BG_RGB), dtype=np.uint8)
    # Subtle coordinate grid (40px spacing clipped nicely)
    for gx in range(80, W, 120):
        cv2.line(canvas, (gx, 72), (gx, H - 56), bgr((13, 20, 33)), 1, cv2.LINE_AA)
    for gy in range(90, H - 50, 120):
        cv2.line(canvas, (64, gy), (W - 64, gy), bgr((13, 20, 33)), 1, cv2.LINE_AA)
    # Rotating subtle orbital arc on top-right
    cx, cy = W - 120, 90
    cv2.circle(canvas, (cx, cy), 340, bgr((18, 30, 50)), 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 250, bgr((22, 42, 68)), 1, cv2.LINE_AA)
    ang = phase_t * 360.0
    cv2.ellipse(canvas, (cx, cy), (250, 250), 0, ang, ang + 65, bgr(ACCENT_RGB), 2, cv2.LINE_AA)
    return canvas


def draw_header_footer(pil_img: Image.Image, step_pill: str, pill_col: tuple, footer_txt: str, global_t: float):
    d = ImageDraw.Draw(pil_img)
    f_mono = load_font(15, bold=True, mono=True)
    f_foot = load_font(14, bold=False, mono=True)

    # Top brand marker
    d.ellipse([(64, 34), (74, 44)], fill=ACCENT_RGB)
    d.text((84, 30), "TERRALENS", font=f_mono, fill=INK_RGB)

    # Step badge on top right
    bw = d.textbbox((0, 0), step_pill, font=f_mono)[2] + 26
    d.rounded_rectangle([(W - 64 - bw, 24), (W - 64, 52)], radius=6, fill=SURFACE_RGB, outline=pill_col, width=1)
    d.text((W - 64 - bw + 13, 30), step_pill, font=f_mono, fill=pill_col)

    # Bottom bar
    d.line([(64, H - 48), (W - 64, H - 48)], fill=(25, 38, 60), width=1)
    d.text((64, H - 36), footer_txt, font=f_foot, fill=MUTED_RGB)

    sec = int(global_t * 30.0)
    sec = min(30, sec)
    d.text((W - 142, H - 36), f"0:{sec:02d} / 0:30", font=f_mono, fill=MUTED_RGB)


# ==============================================================================
# BEAT 1 (0.0s - 5.5s): THE PROBLEM — FAKE CLIMATE TRENDS
# ==============================================================================
def render_beat1(lt: float, gt: float, c: dict) -> np.ndarray:
    cv_img = make_canvas(gt)

    # Animated misleading time series with a sharp 2001 calibration step + El Nino spike
    plot_x0, plot_x1 = 90, W - 90
    plot_y0, plot_y1 = 270, 505
    mid_y = (plot_y0 + plot_y1) // 2

    # Card frame around plot
    cv2.rectangle(cv_img, (64, plot_y0 - 25), (W - 64, plot_y1 + 25), bgr(SURFACE_RGB), -1)
    cv2.rectangle(cv_img, (64, plot_y0 - 25), (W - 64, plot_y1 + 25), bgr((30, 45, 70)), 1, cv2.LINE_AA)

    n_pts = 110
    draw_n = max(2, int(n_pts * ease_out(min(1.0, lt * 1.6))))
    pts = []
    for k in range(draw_n):
        u = k / float(n_pts - 1)
        px = int(plot_x0 + u * (plot_x1 - plot_x0))
        # Autocorrelated wave + sudden calibration jump at u=0.45 + El Nino spike at u=0.38
        step = -48 if u > 0.45 else 24
        spike = -55 * math.exp(-((u - 0.38) / 0.035) ** 2)
        wave = 26 * math.sin(u * 22.0) + 14 * math.cos(u * 41.0)
        py = int(mid_y + step + spike + wave)
        pts.append((px, max(plot_y0, min(plot_y1, py))))

    if len(pts) >= 2:
        for k in range(1, len(pts)):
            col = bgr(AMBER_RGB) if abs(k / n_pts - 0.45) < 0.03 else bgr(ROSE_RGB)
            cv2.line(cv_img, pts[k - 1], pts[k], col, 3, cv2.LINE_AA)

    # Naive linear line appearing false after lt > 0.35
    if lt > 0.32:
        a_line = ease_out((lt - 0.32) / 0.25)
        lx1 = int(plot_x0 + a_line * (plot_x1 - plot_x0))
        ly0 = mid_y + 26
        ly1 = int(ly0 - a_line * 68)
        cv2.line(cv_img, (plot_x0, ly0), (lx1, ly1), bgr(ACCENT_RGB), 3, cv2.LINE_AA)

    # Highlight 2001 sensor jump marker
    if lt > 0.42:
        jx = int(plot_x0 + 0.45 * (plot_x1 - plot_x0))
        cv2.line(cv_img, (jx, plot_y0 - 10), (jx, plot_y1 + 10), bgr(AMBER_RGB), 2, cv2.LINE_AA)
        pulse_r = int(10 + 10 * abs(math.sin(lt * 12.0)))
        cv2.circle(cv_img, (jx, mid_y - 12), pulse_r, bgr(AMBER_RGB), 2, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f_h1 = load_font(54, bold=True)
    f_sub = load_font(22, bold=False)
    f_card_t = load_font(20, bold=True)
    f_card_s = load_font(16, bold=False)
    f_badge = load_font(16, bold=True, mono=True)

    rev = ease_out(min(1.0, lt * 3.0))
    dy = int((1.0 - rev) * 16)
    d.text((64, 78 + dy), c["b1_h1"], font=f_h1, fill=INK_RGB)
    d.text((64, 152 + dy), c["b1_sub"], font=f_sub, fill=MUTED_RGB)

    if lt > 0.45:
        jx = int(plot_x0 + 0.45 * (plot_x1 - plot_x0))
        d.rounded_rectangle([(jx + 10, plot_y0 - 10), (jx + 235, plot_y0 + 22)], radius=5, fill=SURFACE2_RGB, outline=AMBER_RGB, width=1)
        d.text((jx + 20, plot_y0 - 4), c["b1_jump"], font=f_badge, fill=AMBER_RGB)

    # 3 crisp cards along bottom
    cards = c["b1_cards"]
    cw = (W - 128 - 32) // 3
    for idx, (ctitle, csub, ccol) in enumerate(cards):
        appear = ease_out(max(0.0, (lt - 0.18 - idx * 0.14) / 0.25))
        if appear <= 0.01:
            continue
        cx0 = 64 + idx * (cw + 16)
        cy0 = 556 + int((1.0 - appear) * 20)
        d.rounded_rectangle([(cx0, cy0), (cx0 + cw, cy0 + 96)], radius=10, fill=SURFACE_RGB, outline=(34, 48, 72), width=1)
        d.rectangle([(cx0, cy0 + 12), (cx0 + 4, cy0 + 84)], fill=ccol)
        d.text((cx0 + 20, cy0 + 18), ctitle, font=f_card_t, fill=INK_RGB)
        d.text((cx0 + 20, cy0 + 50), csub, font=f_card_s, fill=MUTED_RGB)

    draw_header_footer(pil, c["b1_pill"], ROSE_RGB, c["foot"], gt)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ==============================================================================
# BEAT 2 (5.5s - 11.5s): MULTI-SCALE NASA FUSION (55 km -> 1 km -> 250 m)
# ==============================================================================
def render_beat2(lt: float, gt: float, c: dict) -> np.ndarray:
    cv_img = make_canvas(gt)

    # Left: Animated concentric resolution target (55km -> 1km -> 250m)
    cx, cy = 345, 425
    zoom_t = ease_in_out(min(1.0, lt * 1.25))
    scales = [
        (210, ACCENT_RGB, 1.0),
        (132, AMBER_RGB, max(0.0, (lt - 0.18) / 0.3)),
        (58, EMERALD_RGB, max(0.0, (lt - 0.36) / 0.3)),
    ]
    for radius, col, vis in scales:
        if vis <= 0:
            continue
        r = int(radius * (1.08 - 0.08 * zoom_t))
        # Rounded square footprint look
        cv2.rectangle(cv_img, (cx - r, cy - r), (cx + r, cy + r), bgr(SURFACE_RGB if radius == 210 else SURFACE2_RGB), -1)
        cv2.rectangle(cv_img, (cx - r, cy - r), (cx + r, cy + r), bgr(col), 2, cv2.LINE_AA)
    # Crosshair lines
    cv2.line(cv_img, (cx - 225, cy), (cx + 225, cy), bgr((38, 56, 84)), 1, cv2.LINE_AA)
    cv2.line(cv_img, (cx, cy - 225), (cx, cy + 225), bgr((38, 56, 84)), 1, cv2.LINE_AA)
    pulse = int(8 + 6 * abs(math.sin(lt * 10.0)))
    cv2.circle(cv_img, (cx, cy), pulse, bgr(EMERALD_RGB), -1, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f_h1 = load_font(50, bold=True)
    f_sub = load_font(22, bold=False)
    f_lbl = load_font(16, bold=True, mono=True)
    f_stat = load_font(44, bold=True, mono=True)
    f_desc = load_font(19, bold=False)

    rev = ease_out(min(1.0, lt * 3.0))
    dy = int((1.0 - rev) * 16)
    d.text((64, 78 + dy), c["b2_h1"], font=f_h1, fill=INK_RGB)
    d.text((64, 146 + dy), c["b2_sub"], font=f_sub, fill=MUTED_RGB)

    # Labels inside concentric squares
    d.text((cx - 190, cy - 195), "55 km · NASA POWER (1981–2025)", font=f_lbl, fill=ACCENT_RGB)
    if lt > 0.2:
        d.text((cx - 118, cy - 118), "1 km · Meteomatics", font=f_lbl, fill=AMBER_RGB)
    if lt > 0.38:
        d.text((cx - 46, cy + 22), "250 m MODIS", font=f_lbl, fill=EMERALD_RGB)

    # Right: 3 clean animated metric counters
    stats = [
        (55697, "", c["b2_s1"], ACCENT_RGB, 0.10),
        (107, " nodes", c["b2_s2"], AMBER_RGB, 0.25),
        (45, " yrs", c["b2_s3"], EMERALD_RGB, 0.40),
    ]
    for idx, (target, unit, desc, col, delay) in enumerate(stats):
        prog = ease_out(max(0.0, (lt - delay) / 0.42))
        y0 = 235 + idx * 135
        d.rounded_rectangle([(620, y0), (W - 64, y0 + 112)], radius=10, fill=SURFACE_RGB, outline=(34, 48, 72), width=1)
        d.rectangle([(620, y0 + 14), (625, y0 + 98)], fill=col)
        val = int(target * prog)
        val_str = f"{val:,}{unit}"
        d.text((650, y0 + 16), val_str, font=f_stat, fill=col)
        d.text((650, y0 + 70), desc, font=f_desc, fill=INK_RGB)

    draw_header_footer(pil, c["b2_pill"], ACCENT_RGB, c["foot"], gt)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ==============================================================================
# BEAT 3 (11.5s - 18.5s): ANTI-BIAS AUDIT — FAKE WIPED VS. REAL SIGNAL
# ==============================================================================
def render_beat3(lt: float, gt: float, c: dict) -> np.ndarray:
    cv_img = make_canvas(gt)

    # Left panel: Fake Trend Exposed (Hamed-Rao flattens it!)
    # Right panel: True Warming Locked (Acker Filter keeps slope!)
    pw = (W - 128 - 24) // 2
    panels = [
        (64, ROSE_RGB, EMERALD_RGB),
        (64 + pw + 24, EMERALD_RGB, ACCENT_RGB),
    ]
    for idx, (px0, c1, c2) in enumerate(panels):
        cv2.rectangle(cv_img, (px0, 230), (px0 + pw, 620), bgr(SURFACE_RGB), -1)
        cv2.rectangle(cv_img, (px0, 230), (px0 + pw, 620), bgr((36, 52, 78)), 1, cv2.LINE_AA)

        # Animated slope inside each panel (y: 350..500)
        gx0, gx1 = px0 + 36, px0 + pw - 36
        gy_mid = 435
        wipe = ease_in_out(max(0.0, (lt - 0.30) / 0.40))
        pts_curve = []
        for k in range(55):
            u = k / 54.0
            x = int(gx0 + u * (gx1 - gx0))
            if idx == 0:
                # High persistence wave with fake upward tilt that turns flat gray under filter
                amp = 34 * math.sin(u * 8.0 + 0.4)
                y = int(gy_mid + amp - 18 * (u - 0.5) * (1.0 - wipe))
            else:
                # True monotonic warming
                amp = 18 * math.sin(u * 14.0)
                y = int(gy_mid + 36 - 78 * u + amp * (1.0 - 0.35 * wipe))
            pts_curve.append((x, y))

        for k in range(1, len(pts_curve)):
            col = bgr(MUTED_RGB if (idx == 0 and wipe > 0.5) else c1)
            cv2.line(cv_img, pts_curve[k - 1], pts_curve[k], col, 3, cv2.LINE_AA)

        # Clean linear trend bar showing true verdict
        if idx == 0:
            slope_y_start = int(gy_mid + 20 * (1.0 - wipe))
            slope_y_end = int(gy_mid - 20 * (1.0 - wipe))
            cv2.line(cv_img, (gx0, slope_y_start), (gx1, slope_y_end), bgr(AMBER_RGB if wipe > 0.5 else ROSE_RGB), 3, cv2.LINE_AA)
        else:
            cv2.line(cv_img, (gx0, gy_mid + 38), (gx1, gy_mid - 42), bgr(EMERALD_RGB), 4, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f_h1 = load_font(50, bold=True)
    f_sub = load_font(22, bold=False)
    f_pt = load_font(24, bold=True)
    f_pm = load_font(34, bold=True, mono=True)
    f_ps = load_font(18, bold=False)

    rev = ease_out(min(1.0, lt * 3.0))
    dy = int((1.0 - rev) * 16)
    d.text((64, 78 + dy), c["b3_h1"], font=f_h1, fill=INK_RGB)
    d.text((64, 146 + dy), c["b3_sub"], font=f_sub, fill=MUTED_RGB)

    wipe_done = lt > 0.45
    # Left panel copy
    d.text((98, 254), c["b3_lt"], font=f_pt, fill=INK_RGB)
    d.text((98, 294), c["b3_lm_after"] if wipe_done else c["b3_lm_before"], font=f_pm, fill=AMBER_RGB if wipe_done else ROSE_RGB)
    d.text((98, 548), c["b3_ls"], font=f_ps, fill=MUTED_RGB)

    # Right panel copy
    rx0 = 64 + pw + 24
    d.text((rx0 + 34, 254), c["b3_rt"], font=f_pt, fill=INK_RGB)
    d.text((rx0 + 34, 294), "+0.197 °C / dec", font=f_pm, fill=EMERALD_RGB)
    d.text((rx0 + 34, 548), c["b3_rs"], font=f_ps, fill=INK_RGB)

    draw_header_footer(pil, c["b3_pill"], AMBER_RGB, c["foot"], gt)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ==============================================================================
# BEAT 4 (18.5s - 25.0s): ZERO-LEAKAGE ML & GLOBAL DISCOVERIES
# ==============================================================================
def render_beat4(lt: float, gt: float, c: dict) -> np.ndarray:
    cv_img = make_canvas(gt)

    # Animated Chronological Timeline Split Bar
    bx0, bx1 = 64, W - 64
    by0, by1 = 236, 300
    split_x = int(bx0 + 0.78 * (bx1 - bx0))

    prog_train = ease_out(min(1.0, lt * 2.2))
    prog_test = ease_out(max(0.0, (lt - 0.25) / 0.4))

    cv2.rectangle(cv_img, (bx0, by0), (bx1, by1), bgr(SURFACE_RGB), -1)
    train_fill = int(bx0 + prog_train * (split_x - bx0))
    cv2.rectangle(cv_img, (bx0, by0), (train_fill, by1), bgr((16, 62, 92)), -1)
    cv2.rectangle(cv_img, (bx0, by0), (train_fill, by1), bgr(ACCENT_RGB), 2, cv2.LINE_AA)

    if prog_test > 0:
        test_fill = int(split_x + 6 + prog_test * (bx1 - split_x - 6))
        cv2.rectangle(cv_img, (split_x + 6, by0), (test_fill, by1), bgr((12, 68, 52)), -1)
        cv2.rectangle(cv_img, (split_x + 6, by0), (test_fill, by1), bgr(EMERALD_RGB), 2, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f_h1 = load_font(48, bold=True)
    f_sub = load_font(22, bold=False)
    f_bar = load_font(18, bold=True, mono=True)
    f_big = load_font(56, bold=True, mono=True)
    f_ct = load_font(21, bold=True)
    f_cs = load_font(17, bold=False)

    rev = ease_out(min(1.0, lt * 3.0))
    dy = int((1.0 - rev) * 16)
    d.text((64, 78 + dy), c["b4_h1"], font=f_h1, fill=INK_RGB)
    d.text((64, 142 + dy), c["b4_sub"], font=f_sub, fill=MUTED_RGB)

    d.text((bx0 + 24, by0 + 20), c["b4_train"], font=f_bar, fill=INK_RGB)
    if prog_test > 0.2:
        d.text((split_x + 22, by0 + 20), "2019–2025 TEST", font=f_bar, fill=EMERALD_RGB)

    # Big Future R2 Hero Metric Strip
    r2_prog = ease_out(max(0.0, (lt - 0.2) / 0.5))
    r2_val = 0.9191 * r2_prog
    d.rounded_rectangle([(64, 330), (W - 64, 444)], radius=10, fill=SURFACE_RGB, outline=(36, 52, 78), width=1)
    d.text((94, 354), f"{r2_val:.4f} R²", font=f_big, fill=EMERALD_RGB)
    d.text((430, 358), c["b4_r2_t"], font=f_ct, fill=INK_RGB)
    d.text((430, 392), c["b4_r2_s"], font=f_cs, fill=MUTED_RGB)

    # 3 Global Verified Finding Cards
    cw = (W - 128 - 32) // 3
    for idx, (stat, label, col) in enumerate(c["b4_nodes"]):
        ap = ease_out(max(0.0, (lt - 0.35 - idx * 0.14) / 0.25))
        if ap <= 0.01:
            continue
        cx0 = 64 + idx * (cw + 16)
        cy0 = 478 + int((1.0 - ap) * 18)
        d.rounded_rectangle([(cx0, cy0), (cx0 + cw, cy0 + 150)], radius=10, fill=SURFACE_RGB, outline=col, width=1)
        d.text((cx0 + 24, cy0 + 26), stat, font=load_font(34, bold=True, mono=True), fill=col)
        d.text((cx0 + 24, cy0 + 84), label, font=f_cs, fill=INK_RGB)

    draw_header_footer(pil, c["b4_pill"], EMERALD_RGB, c["foot"], gt)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ==============================================================================
# BEAT 5 (25.0s - 30.0s): CLOSING — TERRALENS & TEAM PROYECTO HAIL MARY
# ==============================================================================
def render_beat5(lt: float, gt: float, c: dict) -> np.ndarray:
    cv_img = np.full((H, W, 3), bgr(BG_RGB), dtype=np.uint8)

    # Clean vector planet + glowing atmospheric horizon emblem
    cx, cy = W // 2, 245
    ring_expand = ease_out(min(1.0, lt * 2.0))
    cv2.circle(cv_img, (cx, cy), int(125 * ring_expand), bgr((22, 42, 68)), 1, cv2.LINE_AA)
    cv2.circle(cv_img, (cx, cy), int(92 * ring_expand), bgr(ACCENT_RGB), 2, cv2.LINE_AA)
    # Stylized trend pulse inside emblem
    e_pts = [
        (cx - 56, cy + 20),
        (cx - 24, cy + 6),
        (cx - 2, cy - 18),
        (cx + 22, cy + 4),
        (cx + 56, cy - 26),
    ]
    for k in range(1, len(e_pts)):
        cv2.line(cv_img, e_pts[k - 1], e_pts[k], bgr(EMERALD_RGB), 4, cv2.LINE_AA)
    cv2.circle(cv_img, e_pts[-1], 6, bgr(ACCENT_RGB), -1, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f_title = load_font(68, bold=True)
    f_sub = load_font(24, bold=False)
    f_team = load_font(20, bold=True)
    f_members = load_font(17, bold=False, mono=True)

    tw = d.textbbox((0, 0), "TerraLens", font=f_title)[2]
    d.text(((W - tw) // 2, 365), "TerraLens", font=f_title, fill=INK_RGB)

    sw = d.textbbox((0, 0), c["b5_sub"], font=f_sub)[2]
    d.text(((W - sw) // 2, 452), c["b5_sub"], font=f_sub, fill=ACCENT_RGB)

    tm = c["b5_team"]
    tmw = d.textbbox((0, 0), tm, font=f_team)[2]
    d.text(((W - tmw) // 2, 505), tm, font=f_team, fill=INK_RGB)

    mem = "Xavier Peña  ·  Bryam Astudillo  ·  Juan Coronel  ·  David Célleri"
    mw = d.textbbox((0, 0), mem, font=f_members)[2]
    d.text(((W - mw) // 2, 539), mem, font=f_members, fill=MUTED_RGB)

    usr = c["b5_users"]
    uw = d.textbbox((0, 0), usr, font=f_members)[2]
    d.text(((W - uw) // 2, 574), usr, font=f_members, fill=INK_RGB)

    lic = c["b5_lic"]
    lw = d.textbbox((0, 0), lic, font=f_members)[2]
    d.text(((W - lw) // 2, 608), lic, font=f_members, fill=EMERALD_RGB)

    draw_header_footer(pil, "NASA SPACE APPS 2026", EMERALD_RGB, c["foot"], gt)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def build_video(lang: str = "en"):
    out_path = os.path.join(BASE_DIR, f"terralens_demo_30s_{lang}.mp4")

    if lang == "es":
        copy = {
            "foot": "Proyecto Hail Mary · NASA POWER · Terra MODIS 250m · NOAA ONI · Apache 2.0",
            "b1_pill": "01 · EL PROBLEMA",
            "b1_h1": "Casi todas las tendencias satelitales mienten.",
            "b1_sub": "Las regresiones simples confunden memoria del suelo y saltos de satélite con cambio climático.",
            "b1_jump": "Salto Sensor 2001 (CERES)",
            "b1_cards": [
                ("Persistencia (r₁)", "Infla falsos p < 0.05", ROSE_RGB),
                ("Salto ERBE -> CERES", "Falso quiebre solar", AMBER_RGB),
                ("Picos El Niño / 55 km", "Ocultan sequía local", ACCENT_RGB),
            ],
            "b2_pill": "02 · DATOS ESPACIALES",
            "b2_h1": "45 años de la NASA en tres escalas reales.",
            "b2_sub": "Del clima global de 55 km al estrés vegetal de 250 m en el suelo.",
            "b2_s1": "Observaciones reales NASA 1981–2025",
            "b2_s2": "Nodos globales + malla de 96 celdas",
            "b2_s3": "Series climáticas + teleconexión ENSO",
            "b3_pill": "03 · CÓMO SE RESOLVIÓ",
            "b3_h1": "Separamos ilusiones de tendencias reales.",
            "b3_sub": "Hamed-Rao + Pettitt + Test de Acker verifican qué señales sobreviven.",
            "b3_lt": "Humedad Paute (Trampa #1)",
            "b3_lm_before": "p = 0.030  (¡Falsa Alerta!)",
            "b3_lm_after": "p = 0.728  (Falso Positivo)",
            "b3_ls": "Descartado al corregir memoria temporal r₁.",
            "b3_rt": "Calentamiento Real Andino",
            "b3_rs": "Sobrevive tras quitar 5% de extremos ML (p < 0.0001).",
            "b4_pill": "04 · ML SIN FUGAS",
            "b4_h1": "Entrenado en el pasado. Probado en el futuro.",
            "b4_sub": "Partición cronológica estricta sin fugas ni sesgo temporal.",
            "b4_train": "1981–2018 TRAIN + CV (5 Pliegues)",
            "b4_r2_t": "Precisión en Futuro Oculto (2019–2024)",
            "b4_r2_s": "HistGradientBoosting sobre 6,912 meses futuros bloqueados",
            "b4_nodes": [
                ("+0.197 °C/déc", "Calentamiento Andino Paute", EMERALD_RGB),
                ("+0.209 vs -0.329", "Dipolo ENSO Sudamericano", ACCENT_RGB),
                ("+0.579 °C/déc", "Amplificación Ártica Alaska", AMBER_RGB),
            ],
            "b5_sub": "Detective Abierto de Tendencias del Sistema Terrestre",
            "b5_team": "Equipo Proyecto Hail Mary · Ecuador · Reto: Trend Detective",
            "b5_users": "Para: Gestores Hídricos · Agrónomos · Investigadores · Educadores",
            "b5_lic": "Código Abierto Apache 2.0  ·  6/6 Pruebas Científicas PASS",
        }
    else:
        copy = {
            "foot": "Proyecto Hail Mary · NASA POWER · Terra MODIS 250m · NOAA ONI · Apache 2.0",
            "b1_pill": "01 · THE PROBLEM",
            "b1_h1": "Most satellite trend lines lie.",
            "b1_sub": "Naive regressions mistake soil memory, sensor switches & El Niño spikes for long-term shifts.",
            "b1_jump": "2001 CERES Sensor Jump",
            "b1_cards": [
                ("Memory Noise (r₁)", "Inflates false p < 0.05", ROSE_RGB),
                ("ERBE -> CERES Splice", "Fake solar step-change", AMBER_RGB),
                ("El Niño & 55 km Blur", "Masks local droughts", ACCENT_RGB),
            ],
            "b2_pill": "02 · MULTI-SCALE NASA",
            "b2_h1": "45 years of NASA data across three scales.",
            "b2_sub": "Zooming from 55 km climate grids down to 1 km terrain and 250 m plant health.",
            "b2_s1": "Real NASA observations (1981–2025)",
            "b2_s2": "Global biomes + 96 regional grid cells",
            "b2_s3": "Full climate + NOAA ENSO record",
            "b3_pill": "03 · THE SOLUTION",
            "b3_h1": "Stripping noise to isolate true change.",
            "b3_sub": "Hamed-Rao persistence guards + Pettitt breakpoints + Acker outlier removal.",
            "b3_lt": "Paute Soil Moisture (Trap #1)",
            "b3_lm_before": "p = 0.030  (Fake Alert!)",
            "b3_lm_after": "p = 0.728  (False Positive)",
            "b3_ls": "Collapses once autocorrelation r₁ is corrected.",
            "b3_rt": "Andean Structural Warming",
            "b3_rs": "Persists after removing top 5% ML extremes (p < 0.0001).",
            "b4_pill": "04 · ZERO-LEAKAGE ML",
            "b4_h1": "Trained on history. Proven on future years.",
            "b4_sub": "Strict chronological split with zero future information leakage.",
            "b4_train": "1981–2018 TRAIN + 5-FOLD CV",
            "b4_r2_t": "Unseen Future Test Accuracy (2019–2024)",
            "b4_r2_s": "HistGradientBoosting evaluated on 6,912 locked future months",
            "b4_nodes": [
                ("+0.197 °C/dec", "Andean Warming (Paute)", EMERALD_RGB),
                ("+0.209 vs -0.329", "South American ENSO Dipole", ACCENT_RGB),
                ("+0.579 °C/dec", "Arctic Polar Amplification", AMBER_RGB),
            ],
            "b5_sub": "Open Earth System Trend Detective",
            "b5_team": "Team Proyecto Hail Mary · Ecuador · Challenge: Trend Detective",
            "b5_users": "Built for: Water Planners · Agronomists · Scientists · Educators",
            "b5_lic": "Apache 2.0 Open Source  ·  6/6 Scientific Tests Verified",
        }

    # Exact 30.00s choreography (900 frames @ 30fps)
    beats = [
        (0, 165, render_beat1),
        (165, 345, render_beat2),
        (345, 555, render_beat3),
        (555, 750, render_beat4),
        (750, 900, render_beat5),
    ]
    transitions = [b[0] for b in beats[1:]]
    xfade = 10  # 0.33s smooth crossfade

    def frame_at(idx: int) -> np.ndarray:
        gt = idx / float(TOTAL_FRAMES - 1)
        for a, b, fn in beats:
            if a <= idx < b:
                lt = (idx - a) / float(max(1, b - a - 1))
                return fn(lt, gt, copy)
        return render_beat5(1.0, 1.0, copy)

    raw_tmp = out_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    for i in range(TOTAL_FRAMES):
        frm = frame_at(i)
        for ts in transitions:
            if ts <= i < ts + xfade:
                alpha = (i - ts) / float(xfade)
                frm = cv2.addWeighted(frm, alpha, frame_at(max(0, ts - 1)), 1.0 - alpha, 0)
                break
        # Sleek 3px orbital progress bar at bottom edge
        bar_w = int((i / float(TOTAL_FRAMES - 1)) * W)
        cv2.rectangle(frm, (0, H - 3), (bar_w, H), bgr(ACCENT_RGB), -1)
        writer.write(frm)

        if lang == "en" and i in (85, 255, 460, 650, 840):
            cv2.imwrite(os.path.join(ASSETS_DIR, f"motion_kf_{i:03d}.png"), frm)

    writer.release()

    # Encode universal H.264 Main Profile (yuv420p + faststart)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg_exe, "-y", "-i", raw_tmp,
            "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
            "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ],
        check=True,
        capture_output=True,
    )
    os.remove(raw_tmp)
    print(f"Built Motion Explainer H.264 {out_path} -> {os.path.getsize(out_path)/(1024*1024):.2f} MB")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    if arg in ("en", "both"):
        build_video("en")
    if arg in ("es", "both"):
        build_video("es")
