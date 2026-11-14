"""
1-Click Launcher for TerraLens v2.1 Web Observatory.
Starts the FastAPI + Interactive Leaflet/Chart.js Observatory on http://127.0.0.1:8085
and opens the default browser automatically.
"""

import os
import sys
import time
import threading
import webbrowser
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8085")


if __name__ == "__main__":
    print("=======================================================================")
    print("  TerraLens v2.1 — Earth System Trend, Breakpoint & Anomaly Detective")
    print("  Opening Interactive Web Observatory at: http://127.0.0.1:8085")
    print("  (API Documentation available at: http://127.0.0.1:8085/docs)")
    print("=======================================================================")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("api:app", host="127.0.0.1", port=8085, reload=False, app_dir=BASE_DIR)
