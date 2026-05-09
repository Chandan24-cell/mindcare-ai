"""
Entrypoint for running MindCare AI inside Docker or local shells.

Usage:
    python app.py

This starts the FastAPI server backed by backend.main:app using the PORT
environment variable, or 7860 when PORT is unset for Docker Spaces.
"""

import os
import uvicorn

DEFAULT_PORT = 7860


def run():
    port = int(os.getenv("PORT", DEFAULT_PORT))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()
