"""
Starts the TerraLens observatory on http://127.0.0.1:8085 and opens it in the default browser.
The API documentation is served at /docs.
"""

import os
import sys
import threading
import time
import webbrowser

import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOST, PORT = "127.0.0.1", 8085
URL = f"http://{HOST}:{PORT}"

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _open_browser() -> None:
    time.sleep(1.5)
    webbrowser.open(URL)


if __name__ == "__main__":
    print(f"TerraLens — Earth System Trend Detective\n  Observatory: {URL}\n  API docs:    {URL}/docs")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("api:app", host=HOST, port=PORT, reload=False, app_dir=BASE_DIR)
