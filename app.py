"""
Entrypoint for running MindCare AI inside Docker or local shells.

Usage:
    python app.py

This starts the FastAPI server backed by backend.main:app on port 8000,
respecting the PORT environment variable when provided by cloud hosts.
"""

import os
import uvicorn


def run():
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()
