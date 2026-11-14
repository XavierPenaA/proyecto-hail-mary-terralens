import os
import sys
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

W, H = 1280, 720
FPS = 30
TOTAL_FRAMES = 900  # Exact 30.00 seconds

# Intentional Atmospheric Palette
BG = (7, 11, 18)
SURFACE = (14, 21, 34)
INK = (241, 245, 249)
MUTED = (148, 163, 184)
ACCENT = (56, 189, 248)
EMERALD = (16, 185, 129)
AMBER = (245, 158, 11)
ROSE = (244, 63, 94)


def load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if mono:
        candidates = [
            r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf",
        ]
    elif bold:
        candidates = [
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ]
    else:
        candidates = [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def prepare_bg_image(img_path: str) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    # Oversize slightly (1380x776) so subtle smooth Ken Burns zoom has crisp detail
    target_w = int(W * 1.08)
    ratio = target_w / im.width
    target_h = int(im.height * ratio)
    im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return im


def render_app_frame(
    bg_img: Image.Image,
    local_t: float,
    tag: str,
    title: str,
    subtitle: str,
    badge_color: tuple,
    pan_dir: int = 1,
) -> np.ndarray:
    eased = local_t
    # Subtle smooth pan/zoom within 1280x720 window
    zoom = 1.0 + 0.04 * (eased if pan_dir > 0 else (1.0 - eased))
    crop_w = int(bg_img.width / zoom)
    crop_h = int(crop_w * H / W)
    max_x = max(0, bg_img.width - crop_w)
    max_y = max(0, bg_img.height - crop_h)
    x0 = int(max_x * (0.2 + 0.6 * eased)) if pan_dir > 0 else int(max_x * (0.8 - 0.6 * eased))
    y0 = int(max_y * 0.15)
    cropped = bg_img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((W, H), Image.Resampling.LANCZOS)

    # Composite sleek dark chyron scrim at bottom
    frame = cropped.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Bottom architectural broadcast bar (high opacity for clean legibility)
    draw.rectangle([(0, H - 122), (W, H)], fill=(7, 11, 18, 250))
    draw.rectangle([(0, H - 124), (W, H - 122)], fill=(*badge_color, 235))

    # Animate chyron entrance in first 0.5s
    slide_in = ease_out_cubic(min(1.0, local_t * 4.5))
    y_shift = int((1.0 - slide_in) * 14)

    f_tag = load_font(15, bold=True, mono=True)
    f_title = load_font(29, bold=True)
    f_sub = load_font(19, bold=False)

    # Tag pill with solid backdrop so background chart text never bleeds through
    tag_bbox = draw.textbbox((0, 0), tag, font=f_tag)
    tag_w = (tag_bbox[2] - tag_bbox[0]) + 28
    draw.rounded_rectangle([(42, H - 106 + y_shift), (42 + tag_w, H - 78 + y_shift)], radius=5, fill=(14, 24, 40, 255), outline=(*badge_color, 235), width=1)
    draw.text((54, H - 101 + y_shift), tag, font=f_tag, fill=(*badge_color, 255))

    # Main title & metric subtitle
    draw.text((42, H - 68 + y_shift), title, font=f_title, fill=(*INK, 255))
    draw.text((42, H - 31 + y_shift), subtitle, font=f_sub, fill=(*MUTED, 255))

    composited = Image.alpha_composite(frame, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


def render_slate(title: str, lead: str, pills: list, footer: str, local_t: float, accent: tuple) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Subtle orbital coordinate arcs
    draw.ellipse([(W - 440, -180), (W + 180, 440)], outline=(30, 58, 95), width=1)
    draw.ellipse([(W - 360, -100), (W + 100, 360)], outline=(56, 189, 248), width=1)
    draw.line([(80, 96), (W - 80, 96)], fill=(30, 41, 59), width=1)
    draw.line([(80, H - 88), (W - 80, H - 88)], fill=(30, 41, 59), width=1)

    f_brand = load_font(18, bold=True, mono=True)
    f_h1 = load_font(54, bold=True)
    f_lead = load_font(24, bold=False)
    f_pill = load_font(17, bold=True, mono=True)

    # Brand header
    draw.ellipse([(82, 56), (94, 68)], fill=accent)
    draw.text((106, 51), "TERRALENS v2.1  ·  NASA SPACE APPS 2026", font=f_brand, fill=INK)

    reveal = ease_out_cubic(min(1.0, local_t * 2.5))
    y_off = int((1.0 - reveal) * 18)

    # Headline
    draw.multiline_text((82, 185 + y_off), title, font=f_h1, fill=INK, spacing=12)
    draw.multiline_text((82, 345 + y_off), lead, font=f_lead, fill=MUTED, spacing=8)

    # Metric pills
    px = 82
    py = 485 + y_off
    for p_txt, p_col in pills:
        bw = len(p_txt) * 11 + 30
        draw.rounded_rectangle([(px, py), (px + bw, py + 42)], radius=6, fill=SURFACE, outline=p_col, width=1)
        draw.text((px + 15, py + 10), p_txt, font=f_pill, fill=p_col)
        px += bw + 14

    draw.text((82, H - 66), footer, font=f_brand, fill=MUTED)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def build_video(lang: str = "en"):
    out_path = os.path.join(BASE_DIR, f"terralens_demo_30s_{lang}.mp4")

    assets = {
        "trends": prepare_bg_image(os.path.join(ASSETS_DIR, "app_trends.png")),
        "acker": prepare_bg_image(os.path.join(ASSETS_DIR, "app_acker.png")),
        "telemetry": prepare_bg_image(os.path.join(ASSETS_DIR, "app_telemetry.png")),
        "map": prepare_bg_image(os.path.join(ASSETS_DIR, "app_map_global.png")),
    }

    if lang == "es":
        copy = {
            "s1_t": "TerraLens:\nDetective de Tendencias Terrestres",
            "s1_l": "¿Las tendencias climáticas de 45 años son reales o ilusiones estadísticas?\nSeparamos calentamiento estructural de saltos satelitales y picos El Niño.",
            "s1_p": [("45 Años NASA POWER", ACCENT), ("ML sin Fugas", EMERALD), ("Malla + 11 Nodos", AMBER)],
            "s1_f": "Proyecto Hail Mary (Ecuador): X. Peña · B. Astudillo · J. Coronel · D. Célleri",
            "s2_tag": "01 · ESTADÍSTICA ANTI-SESGO",
            "s2_t": "Tendencias Históricas: Pendiente Theil-Sen vs. Ruido",
            "s2_s": "Paute +0.197 °C/década (Hamed-Rao p < 0.0001)  ·  Salto Pettitt 2002-11 (+0.53 °C)",
            "s3_tag": "02 · TEST DE ACKER Y QUIBRES",
            "s3_t": "Detectando Falsas Tendencias por Satélite y El Niño",
            "s3_s": "Filtro CERES 2001+ elimina saltos ERBE  ·  Quitar Extremos ML prueba robustez real",
            "s4_tag": "03 · MACHINE LEARNING SIN FUGAS",
            "s4_t": "Split Cronológico 3 Vías en 55,697 Observaciones Reales",
            "s4_s": "Entrenado 1981–2018  ·  Prueba Futura 2019–2024 HistGBM R² = 0.9191",
            "s5_tag": "04 · FUSIÓN 55 km -> 250 m Y SONDA GLOBAL",
            "s5_t": "96 Celdas de Ecuador + 11 Nodos Globales de Referencia",
            "s5_s": "Ártico +0.579 °C/década  ·  MODIS NDVI 250m  ·  Consulta NASA POWER en vivo",
            "s6_t": "Ciencia Climática Abierta\ny Verificada de Extremo a Extremo",
            "s6_l": "6/6 Pruebas Científicas PASS  ·  Observatorio Bilingüe EN/ES\nFastAPI + Scikit-Learn + Leaflet · Listo para Producción y Docker",
            "s6_p": [("6/6 Tests PASS", EMERALD), ("55,697 Obs Reales", ACCENT), ("Proyecto Hail Mary", AMBER)],
            "s6_f": "Apache 2.0 · Fuentes: NASA POWER, MODIS, NOAA, Leaflet, Chart.js  ·  Space Apps 2026",
        }
    else:
        copy = {
            "s1_t": "TerraLens:\nEarth System Trend Detective",
            "s1_l": "Are 45-year satellite climate trends real—or statistical illusions?\nWe separate structural warming from sensor jumps and El Niño spikes.",
            "s1_p": [("45-Year NASA POWER", ACCENT), ("Zero-Leakage ML", EMERALD), ("11 Global Nodes", AMBER)],
            "s1_f": "Proyecto Hail Mary (Ecuador): X. Peña · B. Astudillo · J. Coronel · D. Célleri",
            "s2_tag": "01 · ANTI-BIAS STATISTICS",
            "s2_t": "Long-Term Trends: Robust Theil-Sen & Persistence Guards",
            "s2_s": "Andean Paute +0.197 °C/decade (Hamed-Rao p < 0.0001)  ·  Pettitt Breakpoint 2002-11",
            "s3_tag": "02 · ACKER OUTLIER TEST",
            "s3_t": "Catching Satellite Splices & Isolated El Niño Illusions",
            "s3_s": "CERES 2001+ Guard removes ERBE jumps  ·  Filter ML Extremes re-tests structural slopes",
            "s4_tag": "03 · ZERO-LEAKAGE 3-WAY ML",
            "s4_t": "Chronological 1981–2018 Train vs. 2019–2024 Future Test",
            "s4_s": "55,697 Real NASA Obs  ·  HistGradientBoosting Future Test R² = 0.9191 (RMSE 0.2308)",
            "s5_tag": "04 · MULTI-SCALE 55km -> 250m",
            "s5_t": "96-Cell Ecuador Grid + 11 Global Benchmark Nodes",
            "s5_s": "Arctic Amplification +0.579 °C/dec  ·  MODIS 250m NDVI  ·  Live NASA Global Probe",
            "s6_t": "Open, Audited Earth Science\nReady for Decision-Makers",
            "s6_l": "6/6 Scientific Validity Tests PASS  ·  Bilingual EN/ES Observatory\nFastAPI + Scikit-Learn + Leaflet · Docker & REST API Ready",
            "s6_p": [("6/6 Tests PASS", EMERALD), ("55,697 Real Obs", ACCENT), ("Proyecto Hail Mary", AMBER)],
            "s6_f": "Apache 2.0 · Sources: NASA POWER, MODIS, NOAA, Leaflet, Chart.js  ·  Space Apps 2026",
        }

    # Segment boundaries (in frames, sum = 900 -> 30.00s)
    segments = [
        (0, 108, "s1"),
        (108, 306, "s2"),
        (306, 504, "s3"),
        (504, 684, "s4"),
        (684, 828, "s5"),
        (828, 900, "s6"),
    ]

    # Pre-build frames per frame index
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    f_clock = load_font(15, bold=True, mono=True)
    crossfade_len = 10  # 0.33s crossfade

    def frame_for(idx: int) -> np.ndarray:
        for a, b, kind in segments:
            if a <= idx < b:
                lt = (idx - a) / max(1, (b - a - 1))
                if kind == "s1":
                    return render_slate(copy["s1_t"], copy["s1_l"], copy["s1_p"], copy["s1_f"], lt, ACCENT)
                elif kind == "s2":
                    return render_app_frame(assets["trends"], lt, copy["s2_tag"], copy["s2_t"], copy["s2_s"], ACCENT, 1)
                elif kind == "s3":
                    return render_app_frame(assets["acker"], lt, copy["s3_tag"], copy["s3_t"], copy["s3_s"], AMBER, -1)
                elif kind == "s4":
                    return render_app_frame(assets["telemetry"], lt, copy["s4_tag"], copy["s4_t"], copy["s4_s"], EMERALD, 1)
                elif kind == "s5":
                    return render_app_frame(assets["map"], lt, copy["s5_tag"], copy["s5_t"], copy["s5_s"], ACCENT, -1)
                else:
                    return render_slate(copy["s6_t"], copy["s6_l"], copy["s6_p"], copy["s6_f"], lt, EMERALD)
        return render_slate(copy["s6_t"], copy["s6_l"], copy["s6_p"], copy["s6_f"], 1.0, EMERALD)

    transition_starts = [s[0] for s in segments[1:]]

    for i in range(TOTAL_FRAMES):
        img_cur = frame_for(i)
        # Smooth crossfade across scene boundaries
        for ts in transition_starts:
            if ts <= i < ts + crossfade_len:
                alpha = (i - ts) / float(crossfade_len)
                img_prev = frame_for(max(0, ts - 1))
                img_cur = cv2.addWeighted(img_cur, alpha, img_prev, 1.0 - alpha, 0)
                break

        # Draw progress bar & exact timecode badge at bottom-right
        sec = i // FPS
        tc = f"0:{sec:02d} / 0:30"
        bar_w = int((i / float(TOTAL_FRAMES - 1)) * W)
        cv2.rectangle(img_cur, (0, H - 4), (bar_w, H), (248, 189, 56), -1)  # BGR: cyan-sky #38bdf8

        pil_hud = Image.fromarray(cv2.cvtColor(img_cur, cv2.COLOR_BGR2RGB))
        d_hud = ImageDraw.Draw(pil_hud)
        d_hud.rounded_rectangle([(W - 128, H - 38), (W - 22, H - 14)], radius=4, fill=(14, 21, 34), outline=(51, 65, 85), width=1)
        d_hud.text((W - 117, H - 34), tc, font=f_clock, fill=MUTED)
        final_bgr = cv2.cvtColor(np.array(pil_hud), cv2.COLOR_RGB2BGR)
        writer.write(final_bgr)

        # Save keyframes for visual inspection on EN pass
        if lang == "en" and i in (45, 200, 400, 600, 760, 865):
            kf_path = os.path.join(ASSETS_DIR, f"video_kf_{i:03d}.png")
            cv2.imwrite(kf_path, final_bgr)

    writer.release()

    # Re-encode to universal H.264 (yuv420p + faststart) so browsers, GitHub, and Windows Media Player play seamlessly
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_h264 = out_path + ".h264.mp4"
    subprocess.run(
        [
            ffmpeg_exe, "-y", "-i", out_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            tmp_h264,
        ],
        check=True,
        capture_output=True,
    )
    os.replace(tmp_h264, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Built H.264 {out_path} -> {TOTAL_FRAMES} frames ({TOTAL_FRAMES / FPS:.2f}s), {size_mb:.2f} MB")


if __name__ == "__main__":
    target_lang = sys.argv[1] if len(sys.argv) > 1 else "both"
    if target_lang in ("en", "both"):
        build_video("en")
    if target_lang in ("es", "both"):
        build_video("es")
