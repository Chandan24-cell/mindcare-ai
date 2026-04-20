# MindCare AI – Mental Wellness Analysis System

AI-powered web application that detects facial emotion, infers stress level, and serves actionable wellness guidance with PDF reporting and voice-assisted playback.

---

## Project Overview
- **Purpose:** Early mental-wellness insights using multimodal signals (camera, manual mood, sensor data).
- **Stack:** FastAPI + PyTorch (Vision Transformer), ReportLab, HTML/CSS/JS dashboard with Chart.js and Web Speech API.
- **Model:** Vision Transformer fine-tuned on **FER2013**; weights loaded from `backend/models/vit_small_emotion.pth`.
- **Safety:** Face detection gate prevents emotion inference when no face is visible; backend returns a friendly prompt instead of guessing.

## Features
- Real-time camera & image upload analysis with ViT-based emotion detection.
- Stress level inference from emotion mapping and sensor heuristics.
- Dynamic recommendations: AI-generated when internet/API key is available, rule-based fallback offline (always returns 3 tips).
- PDF report generation and download; email field captured in the request.
- Voice assistant that reads recommendations aloud (toggle in Settings, persisted).
- Dark/light theme toggle with localStorage persistence and accessible contrast.
- Dashboard visualizations (emotion/stress history, confidence) with theme-aware charts.

## Architecture
```
Frontend (HTML/CSS/JS) ──► FastAPI routes (backend/main.py)
                            │
                            ├─ Face detection (backend/face_detection.py)
                            ├─ ViT inference (backend/inference.py + model_loader.py)
                            ├─ Suggestion engine (backend/suggestion_engine.py)
                            └─ PDF generator (backend/report_generator.py)
```

## Model Description
- **Backbone:** `google/vit-base-patch16-224`
- **Fine-tuning:** FER2013 facial emotion dataset, 7 labels (`happy, sad, neutral, angry, fear, disgust, surprise`)
- **Weights:** `backend/models/vit_small_emotion.pth`
- **Emotion → Stress mapping:** happy/surprise → low, neutral/sad → medium, angry/fear/disgust → high.

## Prediction Pipeline (Image)
1. **Image ingest** (camera frame or upload) → Pillow RGB.
2. **Quality checks** (size/brightness).
3. **Face detection** (RetinaFace → MediaPipe → OpenCV Haar). If none: respond with _“No face detected. Please align your face with the camera.”_
4. **Crop + validate** (variance/size to avoid false positives).
5. **ViT preprocessing & inference** → emotion + confidence.
6. **Stress inference** via emotion mapping.
7. **Recommendations** via AI (OpenAI, if reachable) or rule-based fallback (3 items).
8. **Response** to frontend + optional PDF generation.

## API Endpoints
- `GET /` → `frontend/login.html`
- `GET /health` → Service status
- `POST /predict/image?mode=real|mock`  
  Form-data `file` image. Face-gated ViT (real) or mock prediction.
- `POST /predict/manual?mode=real|mock`  
  JSON `{ mood, stress_scale }`
- `POST /predict/sensor?mode=real|mock`  
  JSON `{ heart_rate, hrv, sleep_hours, activity_level, self_mood, stress_scale }`
- `POST /generate-report`  
  JSON `{ email, emotion, stress_level, confidence, suggestions }` → returns `report_path`
- Static mounts: `/frontend/*` assets, `/reports/*` PDF downloads

## Running Locally
1. **Prereqs:** Python 3.11, `backend/models/vit_small_emotion.pth` present.
2. **Install deps**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Start API**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   # or
   python app.py
   ```
4. **Open UI** at `http://localhost:8000/frontend/index.html` (root `/` serves login redirect).
5. **Environment (optional):** set `OPENAI_API_KEY` to enable AI-powered suggestions.

## Running with Docker
```bash
docker build -t mindcare-ai .
docker run -p 8000:8000 mindcare-ai
```
Container entrypoint runs `python app.py` (uvicorn on port 8000). The image installs OpenCV runtime libs needed for face detection.

## Project Structure
```
backend/
  main.py                # FastAPI app & routes
  inference.py           # Face-gated prediction logic
  face_detection.py      # RetinaFace → MediaPipe → Haar pipeline
  model_loader.py        # ViT loader (weights in models/)
  suggestion_engine.py   # AI + rule-based recommendations
  report_generator.py    # PDF creation with ReportLab
frontend/
  index.html             # Dashboard UI
  script.js              # Frontend logic (API calls, chart, voice)
  styles.css             # Theme/readability overrides
reports/                 # Generated PDFs
Dockerfile
requirements.txt
app.py                   # Docker/local entrypoint
```

## How Key Pieces Work
- **Camera input:** WebRTC captures frames → POST `/predict/image` as JPEG blob.
- **Face detection:** RetinaFace (if installed) → MediaPipe → Haar; largest face kept; no face = early 400 response.
- **Emotion prediction:** Cropped face → ViT processor → logits → emotion + confidence → stress mapping.
- **PDF reports:** `backend/report_generator.py` formats results, suggestions, timestamps into `/reports/report_*.pdf`.
- **Voice assistant:** Web Speech API reads recommendations when the Settings toggle is on; state saved in `localStorage`.

## Future Improvements
- Quantization/ONNX export for faster edge inference.
- Add liveness/anti-spoof checks before prediction.
- Multi-language UI copy and voice output.
- Background task to purge old PDF reports securely.
- Expand sensor fusion (EDA/temperature) for richer stress scoring.

---

Built for rapid demos—use results for wellness awareness, not clinical diagnosis.
