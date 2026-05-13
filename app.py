"""
Entrypoint for running MindCare AI inside Docker or local shells.

Usage:
    python app.py

This starts the FastAPI server backed by backend.main:app using the PORT
environment variable, or 7860 when PORT is unset for Docker Spaces.
"""

import os
import uvicorn
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
    host = os.getenv("HOST", "0.0.0.0").strip()
    port = int(os.getenv("PORT", str(DEFAULT_PORT)))

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    run()
