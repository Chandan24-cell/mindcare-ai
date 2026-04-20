# MindCare AI

<div align="center">
  <pre>
███╗   ███╗██╗███╗   ██╗██████╗  ██████╗ █████╗ ██████╗ ███████╗     █████╗ ██╗
████╗ ████║██║████╗  ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██║
██╔████╔██║██║██╔██╗ ██║██║  ██║██║     ███████║██████╔╝█████╗      ███████║██║
██║╚██╔╝██║██║██║╚██╗██║██║  ██║██║     ██╔══██║██╔══██╗██╔══╝      ██╔══██║██║
██║ ╚═╝ ██║██║██║ ╚████║██████╔╝╚██████╗██║  ██║██║  ██║███████╗    ██║  ██║██║
╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚═╝
  </pre>

  <p><strong>Your face tells the story your words never could.</strong></p>
  <p>
    MindCare AI is a production-ready, multi-modal stress detection platform that reads facial emotion in real time,
    maps it to a stress signal, and responds with personalized wellness guidance, downloadable PDF reports,
    and voice-assisted recommendations.
  </p>
  <p>
    Built for rapid demos. Powered by deep learning. Designed for humans.
  </p>
  <p>
    <a href="#quick-start"><strong>Quick Start</strong></a>
    ·
    <a href="#api-reference"><strong>API Docs</strong></a>
    ·
    <a href="#system-architecture"><strong>Architecture</strong></a>
    ·
    <a href="#ml-model--intelligence"><strong>ML Model</strong></a>
    ·
    <a href="#docker-deployment"><strong>Docker</strong></a>
  </p>
</div>

## What Makes This Different

| The Problem | The Solution |
| --- | --- |
| Stress often builds quietly, and self-reporting is delayed, biased, or skipped entirely. Continuous monitoring systems can also be expensive or hard to access. | MindCare AI combines computer vision, manual self-report, and physiological inputs to deliver fast stress estimation with zero dependency on paid inference services. |
| People regularly misjudge their own stress level until symptoms become visible in behavior, performance, or sleep. | A fine-tuned Vision Transformer analyzes face crops in real time, while manual and sensor modes keep the system usable even without camera input. |
| Many demo projects stop at prediction and do not provide a usable experience after inference. | MindCare AI includes recommendations, downloadable reports, charting, voice playback, theme persistence, and a full browser dashboard. |

## Feature Showcase

| Area | Capabilities |
| --- | --- |
| Live camera analysis | WebRTC capture, frame submission, face-gated inference, live confidence output |
| Image upload mode | Static image analysis, drag-and-drop upload, same face-first pipeline as camera mode |
| Manual mood entry | No-camera workflow using self-reported mood and stress scale |
| Physiological sensors | Heart rate, HRV, sleep hours, activity level, and self mood combined into a stress score |
| Recommendations | OpenAI-powered suggestions when configured, deterministic fallback otherwise |
| PDF reports | ReportLab-generated export with email capture and downloadable report path |
| Frontend dashboard | Chart.js history graph, notifications, voice assistant, and theme persistence |
| Authentication | Supabase-powered login page with email/password and Google sign-in flow |

## System Architecture

```text
                           USER (Browser / Mobile)
                 Camera · Upload · Manual Form · Voice UI
                                      |
                                      | HTTP / REST
                                      v
                         FastAPI Backend (Uvicorn)
                   backend/main.py routes + static serving
                       /            /health    /predict/*
                                      |
         +----------------------------+-----------------------------+
         |                            |                             |
         v                            v                             v
  face_detection.py            inference.py                suggestion_engine.py
 RetinaFace (optional)       ViT preprocessing            OpenAI API (optional)
 MediaPipe (optional)        real + mock modes            rule-based fallback
 OpenCV Haar fallback        stress mapping               always returns guidance
         |                            |
         +-------------+--------------+
                       |
                       v
                model_loader.py
        google/vit-base-patch16-224
        weights from backend/models/
                       |
                       v
              report_generator.py
            ReportLab PDF generation
              reports/report_*.pdf
```

## ML Model & Intelligence

### Vision Transformer Pipeline

Unlike a CNN that focuses on local receptive fields, the Vision Transformer treats the face as a sequence of image patches and learns global relationships across the full frame. That helps the model respond to subtle expression patterns and overall facial context.

```text
Input image
   |
   v
Face detection + crop
   |
   v
Resize to 224 x 224 and normalize
   |
   v
Vision Transformer encoder
   |
   v
Softmax over 7 labels
   |
   v
Emotion -> stress mapping
```

### Model Specifications

| Property | Value |
| --- | --- |
| Backbone | `google/vit-base-patch16-224` |
| Fine-tune dataset | FER2013 |
| Classes | `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` |
| Input resolution | `224 x 224` RGB face crop |
| Output | Emotion label, confidence score, mapped stress level |
| Expected weights | `backend/models/vit_small_emotion.pth` |
| Optional override | `MINDCARE_MODEL_PATH=/path/to/vit_small_emotion.pth` |

### Emotion to Stress Mapping

| Emotion | Stress Level |
| --- | --- |
| `happy`, `surprise` | Low |
| `neutral`, `sad` | Medium |
| `angry`, `fear`, `disgust` | High |

### Face Detection Fallback Pipeline

MindCare AI never runs image inference until a face is confirmed.

```text
Input image
   |
   +--> RetinaFace      (optional, best accuracy for uploaded images)
   |
   +--> MediaPipe       (optional, fast webcam-friendly fallback)
   |
   +--> OpenCV Haar     (available in default setup)
   |
   +--> 400 error       "No face detected. Please align your face with the camera."
```

## Multi-Modal Input Pipelines

| Mode | Input | Pipeline | Best Use Case |
| --- | --- | --- | --- |
| Camera | Live browser frames | Face detect -> crop -> ViT -> stress map | Continuous real-time demo |
| Image upload | JPEG / PNG file | Face detect -> crop -> ViT -> stress map | Single-image analysis |
| Manual mood | Mood + stress scale | Rule-based self-report analysis | No camera workflow |
| Sensor input | HR, HRV, sleep, activity, mood | Weighted heuristic scoring | Wearable-style simulation |

## API Reference

### Base URL

- Local API: `http://localhost:8000`
- Swagger UI: `/docs`
- ReDoc: `/redoc`

### Endpoint Map

```text
GET   /                    -> Redirects to /frontend/login.html
GET   /health              -> Standardized health payload

POST  /predict/image       -> Image-based stress prediction
POST  /predict/manual      -> Mood-form stress prediction
POST  /predict/sensor      -> Sensor-based stress prediction
POST  /generate-report     -> PDF report generation

Static /frontend/*         -> Frontend assets
Static /reports/*          -> Generated PDF downloads
```

All `/predict/*` endpoints support `?mode=real` and `?mode=mock`.

### Response Shape

Prediction endpoints return a consistent JSON structure:

```json
{
  "success": true,
  "mode": "real",
  "emotion": "angry",
  "stress_level": "high",
  "confidence": 0.87,
  "suggestion": [
    "Take 5 slow breaths",
    "Step away from the screen briefly",
    "Try a short grounding exercise"
  ],
  "suggestions": [
    "Take 5 slow breaths",
    "Step away from the screen briefly",
    "Try a short grounding exercise"
  ],
  "message": "Real ML: ViT model detected 'angry' with 0.87 confidence"
}
```

`suggestion` is the original field used by the frontend. `suggestions` is also returned for convenience.

### `POST /predict/image`

Request:

```text
multipart/form-data
file=<image>
mode=real | mock
```

Example response:

```json
{
  "success": true,
  "mode": "real",
  "emotion": "neutral",
  "stress_level": "medium",
  "confidence": 0.74,
  "suggestion": [
    "Maintain regular sleep schedule",
    "Engage in light exercise",
    "Journal your thoughts"
  ],
  "suggestions": [
    "Maintain regular sleep schedule",
    "Engage in light exercise",
    "Journal your thoughts"
  ],
  "message": "Real ML: ViT model detected 'neutral' with 0.74 confidence"
}
```

### `POST /predict/manual`

Request:

```json
{
  "mood": "anxious",
  "stress_scale": 7
}
```

### `POST /predict/sensor`

Request:

```json
{
  "heart_rate": 92,
  "hrv": 28,
  "sleep_hours": 5.5,
  "activity_level": 3,
  "self_mood": "tired",
  "stress_scale": 6
}
```

Note: the current backend expects `activity_level` as a numeric score rather than a label such as `"sedentary"`.

### `POST /generate-report`

Request:

```json
{
  "email": "user@example.com",
  "emotion": "neutral",
  "stress_level": "medium",
  "confidence": 0.74,
  "reason": "Manual input analyzed: mood=neutral, stress scale=6/10",
  "suggestions": [
    "Maintain regular sleep schedule",
    "Engage in light exercise",
    "Journal your thoughts"
  ]
}
```

Response:

```json
{
  "status": "success",
  "report_path": "reports/report_20260420_143012.pdf"
}
```

## Project Structure

```text
mindcare-ai/
|
|-- backend/
|   |-- main.py
|   |-- inference.py
|   |-- face_detection.py
|   |-- model_loader.py
|   |-- suggestion_engine.py
|   |-- report_generator.py
|   |-- schemas.py
|   |-- utils/
|   `-- models/
|
|-- frontend/
|   |-- login.html
|   |-- index.html
|   |-- script.js
|   |-- styles.css
|   `-- favicon.svg
|
|-- reports/
|-- Dockerfile
|-- render.yaml
|-- requirements.txt
|-- app.py
`-- README.md
```

## Quick Start

### Prerequisites

| Requirement | Version / Value | Notes |
| --- | --- | --- |
| Python | 3.11+ | Required |
| Model weights | `vit_small_emotion.pth` | Needed for real image mode |
| OpenAI API key | Optional | Enables AI-generated suggestions |
| Docker | Optional | For containerized deployment |

### 1. Clone and Install

```bash
git clone https://github.com/your-username/mindcare-ai.git
cd mindcare-ai

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
export OPENAI_API_KEY=sk-...
export MINDCARE_MODEL_PATH=/absolute/path/to/vit_small_emotion.pth
```

Only `OPENAI_API_KEY` is optional. If `MINDCARE_MODEL_PATH` is unset, the backend looks for the model in `backend/models/vit_small_emotion.pth`.

### 3. Start the Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# or
python app.py
```

### 4. Open the App

- Login page: `http://localhost:8000/`
- Dashboard: `http://localhost:8000/frontend/index.html`
- Swagger docs: `http://localhost:8000/docs`

## Docker Deployment

```bash
docker build -t mindcare-ai .

docker run -p 8000:8000 mindcare-ai

docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  mindcare-ai

docker run -p 8000:8000 \
  -v "$(pwd)/reports:/app/reports" \
  mindcare-ai
```

The Docker image uses `python:3.11-slim`, installs OpenCV runtime libraries, and starts the API with `python app.py`.

## Render Deployment

This repository already includes [`render.yaml`](render.yaml), configured to:

- use Python 3.11
- install dependencies with `pip install -r requirements.txt`
- start the backend with `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

## Frontend Dashboard

The browser UI is a single-page dashboard backed by a dedicated login screen.

| Panel | Description |
| --- | --- |
| Live camera | Webcam preview, frame capture, and real-time analysis |
| Image upload | Drag-and-drop or picker-based static image inference |
| Manual mood form | Mood and stress self-report flow |
| Sensor input | Biometrics and activity scoring form |
| Charts | History of stress and confidence in Chart.js |
| Notifications | Toast-style feedback for success, failure, and warnings |
| Settings | Theme toggle, voice toggle, notification toggle |
| Voice assistant | Web Speech API playback of recommendations |

## Technology Stack

| Layer | Technology | Why It Fits |
| --- | --- | --- |
| Frontend | HTML5, CSS3, Vanilla JS | Fast to ship, lightweight, no framework runtime |
| Auth | Supabase Auth | Login flow with email/password and Google OAuth |
| Charts | Chart.js | Lightweight and browser-friendly |
| Voice | Web Speech API | Native browser speech synthesis |
| Backend | FastAPI + Uvicorn | Async API server with built-in docs |
| ML | PyTorch + Transformers | Standard stack for ViT-based inference |
| Face detection | RetinaFace, MediaPipe, OpenCV | Accuracy-to-speed fallback chain |
| Reports | ReportLab | Pure Python PDF generation |
| Recommendations | OpenAI API + rule engine | Better personalization with reliable offline fallback |
| Deployment | Docker, Render | Portable local and cloud execution |

## Roadmap

### v1.0

- Real-time ViT inference
- Multi-modal input support
- PDF reporting
- Voice-assisted UI

### v1.1

- ONNX export for edge inference
- Liveness and anti-spoof checks before prediction

### v2.0

- Multi-language UI and voice output
- Expanded sensor fusion
- Automated PDF cleanup task

### v3.0

- CI/CD hardening
- Mobile-first PWA
- Wearable integrations

## Important Notes

- Privacy: image processing runs on your own server by default, and recommendations fall back locally when OpenAI is unavailable.
- Medical disclaimer: MindCare AI is a wellness-awareness tool, not a medical device or diagnostic system.
- Model behavior: facial emotion recognition is probabilistic and should be treated as a signal, not ground truth.
- Optional detectors: RetinaFace and MediaPipe are not installed in the base `requirements.txt`; the default setup still works with OpenCV Haar fallback.

## Author

<div align="center">
  <strong>Chandan Kumar Sah</strong><br>
  Department of Artificial Intelligence &amp; Machine Learning<br>
  Computer Vision · Deep Learning · MLOps<br>
  PyTorch · TensorFlow · OpenCV · Docker · CI/CD
  <br><br>
  MindCare AI · v1.0 · MIT License
</div>

If this project helped you, consider giving it a star on GitHub.
