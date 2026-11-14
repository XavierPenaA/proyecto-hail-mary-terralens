import os
from playwright.sync_api import sync_playwright

OUT = r"c:\Users\xavit\Desktop\Sebas\Nasa\deliverables\assets"
ARTIFACTS = r"C:\Users\xavit\.gemini\antigravity\brain\78f8de84-6f79-4485-9732-4dc872a37ae4"
os.makedirs(OUT, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1520, "height": 900}, device_scale_factor=2)
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)

    # 1. Load app cleanly in EN
    page.goto("http://127.0.0.1:8085/?tab=tab-trends", wait_until="domcontentloaded", timeout=45000)
    page.wait_for_function("() => { const el = document.getElementById('kpiSlope'); return el && el.textContent.trim() !== '—'; }", timeout=45000)
    page.wait_for_timeout(900)
    page.screenshot(path=os.path.join(OUT, "app_trends.png"))

    # 2. Capture Acker view
    page.click('[data-tab="tab-acker"]')
    page.wait_for_timeout(900)
    page.screenshot(path=os.path.join(OUT, "app_acker.png"))

    # 3. Capture Telemetry view
    page.click('[data-tab="tab-telemetry"]')
    page.wait_for_selector("#cvLeaderboardBody tr", timeout=25000)
    page.wait_for_timeout(900)
    page.screenshot(path=os.path.join(OUT, "app_telemetry.png"))

    # 4. Capture Dipole view
    page.click('[data-tab="tab-dipole"]')
    page.wait_for_timeout(3500)
    page.screenshot(path=os.path.join(OUT, "app_dipole.png"))

    # 5. Capture Multiscale Map view (Global zoom)
    page.click('[data-tab="tab-multiscale"]')
    page.wait_for_selector("#stacTableBody tr", timeout=45000)
    page.wait_for_timeout(700)
    page.click("#btnZoomGlobal")
    page.wait_for_timeout(1200)
    page.screenshot(path=os.path.join(OUT, "app_map_global.png"))

    # 6. Capture About ("Project & Build") in EN and ES
    page.click('[data-tab="tab-about"]')
    page.wait_for_timeout(700)
    page.screenshot(path=os.path.join(OUT, "app_about.png"))

    page.click("#btnLangToggle")
    page.wait_for_timeout(700)
    page.screenshot(path=os.path.join(ARTIFACTS, "terralens_v2_es_about.png"), full_page=True)

    # 7. Export 7-Slide Pitch PDFs (EN + ES)
    deck_url = "file:///c:/Users/xavit/Desktop/Sebas/Nasa/deliverables/terralens_pitch_7_slides.html"
    pdf_page = b.new_page(viewport={"width": 1280, "height": 720})
    pdf_page.goto(deck_url, wait_until="networkidle", timeout=30000)
    pdf_page.emulate_media(media="print")
    pdf_page.pdf(
        path=r"c:\Users\xavit\Desktop\Sebas\Nasa\deliverables\terralens_pitch_7_slides_en.pdf",
        width="13.333in",
        height="7.5in",
        print_background=True,
        margin={"top": "0in", "right": "0in", "bottom": "0in", "left": "0in"},
    )
    pdf_page.goto(deck_url + "?lang=es", wait_until="networkidle", timeout=30000)
    pdf_page.emulate_media(media="print")
    pdf_page.pdf(
        path=r"c:\Users\xavit\Desktop\Sebas\Nasa\deliverables\terralens_pitch_7_slides_es.pdf",
        width="13.333in",
        height="7.5in",
        print_background=True,
        margin={"top": "0in", "right": "0in", "bottom": "0in", "left": "0in"},
    )
    pdf_page.emulate_media(media="screen")
    pdf_page.goto(deck_url, wait_until="networkidle", timeout=30000)
    pdf_page.screenshot(path=os.path.join(ARTIFACTS, "deck_slide1_preview.png"), clip={"x": 0, "y": 28, "width": 1280, "height": 720})

    print("ALL_DONE ERRORS:", errs)
    b.close()
