from pathlib import Path
import gdown

MODEL_DIR = Path("backend/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "vit_small_emotion.pth"

FILE_ID = "1N25XPR-GrmyhxtonGee6DC-lMIKsSQrr"

URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not MODEL_PATH.exists():
    print("Downloading model...")
    gdown.download(URL, str(MODEL_PATH), quiet=False)
    print("Model downloaded successfully.")
else:
    print("Model already exists.")
