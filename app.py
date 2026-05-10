"""
Entrypoint for running MindCare AI inside Docker or local shells.

Usage:
    python app.py

This starts the FastAPI server backed by backend.main:app using the PORT
environment variable, or 7860 when PORT is unset for Docker Spaces.
"""

import os
import uvicorn
import webbrowser
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path, override=True)
print(".env path =", dotenv_path)
print("MAIL_SENDER =", os.getenv("MAIL_SENDER"))
print("MAIL_PASSWORD =", os.getenv("MAIL_PASSWORD"))

DEFAULT_PORT = 7860


def run():
    port = int(os.getenv("PORT", DEFAULT_PORT))
    environment = os.getenv("ENVIRONMENT", "development").strip().lower()
    host = os.getenv(
        "HOST",
        "127.0.0.1" if environment == "development" else "0.0.0.0",
    ).strip()

    browser_url = (
        f"http://localhost:{port}"
        if host == "0.0.0.0"
        else f"http://{host}:{port}"
    )

    # Auto-open browser only for local development
    if environment == "development":
        try:
            webbrowser.open(browser_url)
        except Exception:
            pass  # Silently fail if browser opening fails

    uvicorn.run("backend.main:app", host=host, port=port)


if __name__ == "__main__":
    run()
