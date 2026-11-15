"""
Captures the observatory screenshots used by the pitch deck and exports the deck to PDF (EN and ES).

Start the app first (python run_terralens.py), then:
    pip install playwright && python -m playwright install chromium
    python deliverables/capture_and_build_all.py [--base-url http://127.0.0.1:8085]

Set CHROMIUM_PATH to use an existing Chromium binary instead of Playwright's download.
"""

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

DELIVERABLES = Path(__file__).resolve().parent
ASSETS = DELIVERABLES / "assets"
DECK = DELIVERABLES / "terralens_pitch_7_slides.html"
IDLE = "() => document.getElementById('loadingIndicator').style.display === 'none'"
# The observatory animates its counters and charts; wait until they stop before the screenshot.
ANIMATION_SETTLE_MS = 2600

# screenshot file -> (query string, ready element, panel to frame, animation wait in ms)
SHOTS = {
    "app_trends.png": ("?lang=en&theme=dark&region=PE_Altiplano", "#answerText:not(:empty)", "#answerCard", 600),
    "app_acker.png": ("?lang=en&theme=dark&region=BR_Pantanal&tab=tab-acker", "#ackerTableBody tr", "#tab-acker", 800),
    "app_telemetry.png": ("?lang=en&theme=dark&tab=tab-telemetry", "#cvLeaderboardBody tr", "#tab-telemetry", 800),
    "app_dipole.png": ("?lang=en&theme=dark&tab=tab-dipole", "#dipoleSummaryStrip .kpi-card", "#tab-dipole", 1500),
    "app_map_global.png": ("?lang=en&theme=dark&tab=tab-multiscale", "#stacTableBody tr", "#mapPanel", 2500),
    "app_about.png": ("?lang=en&theme=dark&tab=tab-about", "#tab-about.active", "#tab-about", 500),
}


def capture(browser, base_url: str) -> list:
    errors = []
    page = browser.new_page(viewport={"width": 1520, "height": 900}, device_scale_factor=2)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    for name, (query, selector, panel, wait_ms) in SHOTS.items():
        page.goto(f"{base_url}/{query}", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector(selector, timeout=120000)
        page.wait_for_function(IDLE, timeout=60000)
        page.locator(panel).evaluate("(panel) => panel.scrollIntoView({block: 'start', behavior: 'instant'})")
        page.wait_for_timeout(wait_ms + ANIMATION_SETTLE_MS)
        page.screenshot(path=str(ASSETS / name))
        print("captured", name)
    page.close()
    return errors


def export_deck(browser) -> None:
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    for lang in ("en", "es"):
        page.goto(f"{DECK.as_uri()}?lang={lang}", wait_until="networkidle", timeout=60000)
        page.emulate_media(media="print")
        out = DELIVERABLES / f"terralens_pitch_7_slides_{lang}.pdf"
        page.pdf(path=str(out), width="13.333in", height="7.5in", print_background=True,
                 margin={"top": "0in", "right": "0in", "bottom": "0in", "left": "0in"})
        print("exported", out.name)
    page.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8085")
    args = parser.parse_args()
    ASSETS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        errors = capture(browser, args.base_url.rstrip("/"))
        export_deck(browser)
        browser.close()
    print("console errors:", errors or "none")


if __name__ == "__main__":
    main()
