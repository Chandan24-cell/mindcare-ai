---
title: MindCare AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
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

  <p>Built for rapid demos. Powered by deep learning. Designed for humans.</p>

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

---

## What Makes This Different

| Traditional Wellness Apps | MindCare AI |
| --- | --- |
| Depend heavily on manual self-reporting | Uses multi-modal AI fusion across facial, behavioral, physiological, and contextual signals |
| Limited emotional understanding | Real-time emotion recognition with ViT-based affective inference |
| Stateless interactions | Longitudinal session memory, behavioral trend tracking, and adaptive profiling |
| Generic wellness tips | Personalized recommendations powered by contextual reasoning engines |
| No explainability | Transparent explainability, modality reliability, and reasoning traces |
| No predictive intelligence | Future-state simulation, burnout forecasting, fatigue analysis, and cognitive load estimation |
| Usually frontend demos only | Enterprise-grade backend architecture with observability, health monitoring, websocket streaming, and platform APIs |
| Single-mode input systems | Camera, upload, manual, sensor, and realtime streaming pipelines |

---

## Feature Showcase

| Area | Capabilities |
| --- | --- |
| Live camera analysis | WebRTC capture, frame submission, face-gated inference, live confidence output |
| Image upload mode | Static image analysis, drag-and-drop upload, same face-first pipeline as camera mode |
| Manual mood entry | No-camera workflow using self-reported mood and stress scale |
| Physiological sensors | Heart rate, HRV, sleep hours, activity level, and self mood combined into a stress score |
| Recommendations | DeepSeek-powered suggestions when configured, deterministic fallback otherwise |
| PDF reports | ReportLab-generated export with email capture and downloadable report path |
| Frontend dashboard | Chart.js history graph, notifications, voice assistant, and theme persistence |

---

## System Architecture

```text
                           USER INTERACTION LAYER
         Camera · Upload · Manual Input · Sensors · WebSocket Stream
                                      |
                                      v
                          FRONTEND DASHBOARD (SPA)
                Charts · Voice UI · Reports · Theme Engine · Settings
                                      |
                                      v
                           FASTAPI APPLICATION CORE
                     REST APIs · WebSocket APIs · Static Serving
                                      |
        +-----------------------------+------------------------------+
        |                             |                              |
        v                             v                              v
  Prediction Engine            Intelligence Layer           Platform Layer
  ViT Inference                Analytics & Cognition        Enterprise APIs
        |                             |                              |
        |                             |                              |
        v                             v                              v
  Face Detection              Trend Analytics               Auth Scaffolding
  MediaPipe                   Burnout Detection             Metrics
  OpenCV                      Explainability                Monitoring
  RetinaFace                  Behavioral Profiling          Health Checks
        |                     Risk Forecasting              Queue Systems
        |                     Fatigue Detection             WebSockets
        |                     Personalization               Structured Logging
        |                     Future Simulation             MLOps Scaffolding
        +-----------------------------+------------------------------+
                                      |
                                      v
                             REPORTING & EXPORTS
                     PDF Reports · Research Export APIs
```

---

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

---

## Advanced Intelligence Layer

MindCare AI evolved from a simple emotion detector into a research-oriented cognitive wellness platform.

The system now includes deterministic intelligence engines for:

| Intelligence System | Purpose |
| --- | --- |
| Multimodal Fusion Engine | Combines image, manual, and sensor signals into unified wellness inference |
| Trend Analytics Engine | Tracks emotional progression and longitudinal stress trends |
| Burnout Detection | Identifies sustained high-risk emotional patterns |
| Stability & Recovery Analysis | Measures emotional consistency and resilience recovery |
| Explainability Engine | Produces transparent reasoning traces and modality confidence |
| Behavioral Profiling | Learns user behavioral tendencies over time |
| Adaptive Recommendation Engine | Personalizes interventions based on historical effectiveness |
| Emotional Drift Engine | Detects gradual emotional movement across sessions |
| Risk Forecasting Engine | Projects future wellness risk trajectories |
| Cognitive Load Estimation | Estimates overload and mental fatigue |
| Intervention Simulation | Simulates possible future wellness outcomes |
| Temporal Reasoning Engine | Understands longitudinal emotional transitions |
| Self-Evolving Profiles | Continuously adapts contextual user understanding |
| Orchestration Engine | Coordinates intelligence outputs into unified insights |

All systems are additive, modular, and production-safe.

---

## Multi-Modal Input Pipelines

| Mode | Input | Pipeline | Best Use Case |
| --- | --- | --- | --- |
| Camera | Live browser frames | Face detect -> crop -> ViT -> stress map | Continuous real-time demo |
| Image upload | JPEG / PNG file | Face detect -> crop -> ViT -> stress map | Single-image analysis |
| Manual mood | Mood + stress scale | Rule-based self-report analysis | No camera workflow |
| Sensor input | HR, HRV, sleep, activity, mood | Weighted heuristic scoring | Wearable-style simulation |

---

## API Reference

### Base URL

- Local API: `http://localhost:8000`
- Swagger UI: `/docs`
- ReDoc: `/redoc`

### Endpoint Map

```text
GET   /                    -> Frontend dashboard entry page
GET   /health              -> Lightweight runtime health status

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

> `suggestion` is the original field used by the frontend. `suggestions` is also returned for convenience.

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
  "self_mood": "neutral",
  "stress_scale": 6
}
```

> `activity_level` is a numeric score from 1 (sedentary) to 10 (very active).

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

---

## Research & Simulation APIs

| Endpoint | Purpose |
| :--- | :--- |
| `/research/benchmarks` | Research benchmarking |
| `/research/explainability` | Explainability exports |
| `/research/simulation` | Intervention simulation |
| `/research/export/json` | JSON export |
| `/research/export/csv` | CSV export |
| `/research/export/markdown` | Markdown export |

> Please refer to the specific API documentation for details on required parameters, headers, and request/response payloads for each endpoint.

---

## Project Structure

```text
mindcare-ai/
├── backend/
│   ├── api/
│   ├── analytics/
│   ├── cognition/
│   ├── copilot/
│   ├── infrastructure/
│   ├── intelligence/
│   ├── middleware/
│   ├── ml/
│   ├── mlops/
│   ├── observability/
│   ├── orchestration/
│   ├── platform/
│   ├── realtime/
│   ├── realtime_streaming/
│   ├── research/
│   ├── security/
│   ├── simulation/
│   ├── utils/
│   ├── main.py
│   └── models/
├── frontend/
│   ├── css/
│   ├── js/
│   ├── assets/
│   └── index.html
├── reports/
├── deployment/
├── docs/
├── Dockerfile
├── requirements.txt
├── app.py
└── README.md
```

### Architecture Notes

- `backend/main.py` owns FastAPI route registration and static file mounting.
- `backend/api/request_utils.py` keeps request payload diagnostics and validation formatting out of route bodies.
- `backend/inference.py`, `backend/model_loader.py`, and `backend/face_detection.py` preserve the ML pipeline and are intentionally not coupled to frontend concerns.
- `frontend/js/app.js` owns prediction workflows: API calls, camera capture, manual/sensor forms, results, charts, reports, and toasts.
- `frontend/js/dashboard-ui.js` owns non-prediction dashboard chrome: greetings, settings persistence, theme toggles, native notifications, and voice assistant.
- `frontend/styles.css` and `frontend/script.js` remain as compatibility paths so older cached pages and static URLs continue to work.

---

## Quick Start

### Prerequisites

| Requirement | Version / Value | Notes |
| --- | --- | --- |
| Python | 3.11+ | Required |
| Model weights | `vit_small_emotion.pth` | Needed for real image mode |
| DeepSeek API key | Optional | Enables AI-generated suggestions |
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
export OPENROUTER_API_KEY=your_api_key_here
export MINDCARE_MODEL_PATH=/absolute/path/to/vit_small_emotion.pth
```

> Only `OPENROUTER_API_KEY` is optional. If `MINDCARE_MODEL_PATH` is unset, the backend looks for the model in `backend/models/vit_small_emotion.pth`.

### 3. Start the Server

```bash
PORT=8000 python app.py

# or for LAN/dev container binding
HOST=0.0.0.0 PORT=8000 uvicorn backend.main:app --reload
```

> For local browser access, open `http://localhost:8000` or `http://127.0.0.1:8000`.
> `0.0.0.0` is only a bind address and should not be entered into the browser.

### 4. Open the App

- Dashboard: `http://localhost:8000/`
- Direct dashboard page: `http://localhost:8000/frontend/index.html`
- Swagger docs: `http://localhost:8000/docs`

---

## Docker Deployment

```bash
docker build -t mindcare-ai .

docker run -p 7860:7860 mindcare-ai

docker run -p 7860:7860 \
  -e OPENROUTER_API_KEY=your_api_key_here \
  mindcare-ai

docker run -p 7860:7860 \
  -v "$(pwd)/reports:/app/reports" \
  mindcare-ai
```

The Docker image uses `python:3.11-slim`, installs OpenCV runtime libraries, and starts the API with `python app.py`. The launcher binds to the `PORT` environment variable when provided and otherwise uses port `7860`, matching the Hugging Face Docker Space `app_port` metadata.

---

## Render Deployment

This repository already includes [`render.yaml`](render.yaml), configured to:

- use Python 3.11
- install dependencies with `pip install -r requirements.txt`
- start the backend with `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

---

## Frontend Dashboard

The browser UI opens directly to a single-page dashboard.

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

---

## Technology Stack

| Layer | Technology | Why It Fits |
| --- | --- | --- |
| Frontend | HTML5, CSS3, Vanilla JS | Fast to ship, lightweight, no framework runtime |
| Charts | Chart.js | Lightweight and browser-friendly |
| Voice | Web Speech API | Native browser speech synthesis |
| Backend | FastAPI + Uvicorn | Async API server with built-in docs |
| ML | PyTorch + Transformers | Standard stack for ViT-based inference |
| Face detection | RetinaFace, MediaPipe, OpenCV | Accuracy-to-speed fallback chain |
| Reports | ReportLab | Pure Python PDF generation |
| Recommendations | DeepSeek API + rule engine | Better personalization with reliable offline fallback |
| Deployment | Docker, Render | Portable local and cloud execution |

---

## Research Direction

MindCare AI is being expanded beyond traditional emotion recognition into a broader cognitive wellness intelligence platform.

Current research directions include:

- Multimodal affective computing
- Cognitive state modeling
- Emotional memory systems
- Behavioral graph analysis
- Longitudinal emotional forecasting
- Adaptive intervention learning
- Human-centered explainable AI
- Realtime emotional drift detection
- Cognitive overload estimation
- Personalized wellness simulation

The architecture is intentionally modular to support future publication-oriented experimentation and enterprise deployment.

---

## Roadmap

### Phase 1 — Foundation ✅
- ViT-based facial emotion inference
- Multi-input stress analysis
- PDF reporting
- Frontend dashboard

### Phase 2 — Analytics ✅
- Session memory
- Trend analytics
- Burnout detection
- Recovery analysis

### Phase 3 — Intelligence ✅
- Explainability engine
- Behavioral profiling
- Personalization
- Adaptive recommendations

### Phase 4 — Realtime Systems ✅
- Emotional drift analysis
- Fatigue detection
- Cognitive load estimation
- Risk forecasting
- Wellness copilot scaffolding

### Phase 5 — Enterprise Infrastructure ✅
- Health monitoring
- Metrics endpoints
- Structured logging
- MLOps scaffolding
- Queue systems
- WebSocket infrastructure

### Phase 6 — Research Intelligence ✅
- Temporal reasoning
- Emotional memory
- Intervention simulation
- Self-evolving profiles
- Research export systems

### Phase 7 — Platform Expansion 🚧
- Persistent database layer
- Authentication hardening
- Production observability
- User platform APIs

### Future Vision
- ONNX/TensorRT optimization
- Edge AI deployment
- Wearable integration
- Federated wellness learning
- Mobile companion app
- Research publication pipeline

---

## Important Notes

### Privacy & Data Handling

- MindCare AI processes inference locally/on your own deployment infrastructure by default.
- No biometric data is permanently stored unless persistence features are explicitly enabled.
- Session analytics and behavioral profiling are designed with privacy-aware architecture principles.
- Optional AI-enhanced recommendation systems gracefully fall back to deterministic local reasoning when external AI providers are unavailable.

### Medical Disclaimer

MindCare AI is an AI-powered wellness and cognitive analytics platform intended for educational, research, and wellness-monitoring purposes only.

It is **not**:
- a medical device,
- a psychiatric diagnostic system,
- a clinical decision-making tool,
- or a replacement for licensed healthcare professionals.

Always seek qualified medical or mental health support for clinical concerns.

### Model & AI Behavior

- Emotion recognition is probabilistic and inference-based.
- Facial expressions alone cannot determine psychological condition with certainty.
- Multi-modal outputs should be interpreted as supportive wellness signals rather than objective truth.
- Predictions may vary depending on lighting, pose, camera quality, physiological variance, and user context.

### Optional Dependencies

Some advanced modules are optional and intentionally decoupled from the base installation:

| Dependency | Purpose |
| --- | --- |
| RetinaFace | High-accuracy face detection |
| MediaPipe | Fast realtime landmark detection |
| OpenRouter / LLM APIs | AI-generated recommendations |
| WebSockets | Realtime streaming infrastructure |
| ONNX Runtime | Optimized inference acceleration |

The default installation continues to work with OpenCV Haar Cascade fallback detection.

---

## Author

<div align="center">

### Chandan Kumar Sah

Department of Artificial Intelligence & Machine Learning

Computer Vision · Deep Learning · Affective AI · MLOps · Cognitive Intelligence Systems

PyTorch · FastAPI · Vision Transformers · OpenCV · Docker · WebSockets · Explainable AI

<br>

### MindCare AI

Production-Ready Cognitive Wellness Intelligence Platform

Built with:
FastAPI · PyTorch · Vision Transformers · WebSockets · MLOps · Explainable AI

Built for:
Multimodal Intelligence · Realtime Emotion AI · Behavioral Analytics · Wellness Research

<br>

MIT License

</div>

---

# 📄 Report

[![Report](https://img.shields.io/badge/View-Report-blue?style=for-the-badge&logo=google-docs)](https://docs.google.com/document/d/155UIDUjc92SQdJFLGwub_aa_Nt3nSlRYY4k3Nyyjrws/edit?usp=sharing)

---

## 📊 Project Presentation

[![View Presentation](https://img.shields.io/badge/📄_VIEW-PRESENTATION-4285F4?style=for-the-badge)](https://docs.google.com/presentation/d/18mKcJGjFfobiUHiz40zihTVB9IgwRl7c/edit?usp=sharing)

---
## 📑 Project Documentation

[![Documentation](https://docs.google.com/document/d/1bQS9gL9mKowVCHfjjEqp1-MM3Z4O_AAUnR5DtPe8p7o/edit?usp=sharing)

---

## 📄 Research Paper

[![Research Paper](https://img.shields.io/badge/View-Research%20Paper-blue?style=for-the-badge&logo=google-docs)](https://docs.google.com/document/d/1O_FoeEs6L_TmMCifohDw2hWerZxJ1INwOJGgv5gmWNY/edit?usp=sharing)

---

<div align="center">

⭐ If this project helped you, consider starring the repository on GitHub.

</div>
