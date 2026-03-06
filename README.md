# MindCare - AI Stress & Mental State Detection System
MindCare AI — Stress Level Prediction System

AI-powered mental wellness platform that analyzes facial emotion, behavioral input, and physiological signals to estimate a user’s stress level in real time.

The system combines computer vision, machine learning, and user-reported data to provide insights into mental well-being.

Project Overview

MindCare AI is designed to demonstrate how artificial intelligence can assist in early stress detection.

The platform supports multiple input methods:

• Live camera analysis
• Image upload
• Manual mood input
• Physiological sensor data

The goal is to create a multi-modal stress assessment system similar to what real digital health platforms use.

Key Features
Real-Time Emotion Detection

Uses a Vision Transformer model trained on the FER2013 dataset to detect facial emotions.

Stress Level Estimation

Maps detected emotions and physiological indicators to a stress level classification:

Low
Medium
High

Multi-Input System

The platform accepts:

• Live camera feed
• Uploaded images
• Manual emotional input
• Sensor-based metrics

Face Detection Pipeline

Before prediction, the system validates face presence using:

RetinaFace
MediaPipe
OpenCV Haar Cascade

This prevents false predictions.

Interactive Dashboard

Modern UI that displays:

• Detected emotion
• Stress level
• Confidence score
• System notifications

Notification System

Real-time UI alerts for:

• Camera issues
• Missing input
• Face detection failure
• System status updates

External Information Widgets

Dashboard integrations include:

Spotify player
Tech Monitor (AI / Startups / Cybersecurity news)
World Monitor (Global events / geopolitics)

System Architecture

Frontend
User interface, camera input, dashboard visualization.

Backend
FastAPI service handling API endpoints, inference, and data processing.

Machine Learning Layer
Emotion recognition model and stress analysis logic.

Frontend  →  FastAPI Backend  →  ML Inference  →  Response → Dashboard
Project Structure
stress-level-prediction-main

backend/
│
├── main.py
API routes and application startup
│
├── inference.py
ML prediction logic and analysis functions
│
├── model_loader.py
Loads trained model and processor
│
├── schemas.py
API request and response validation
│
└── suggestion_engine.py
Generates wellness suggestions

frontend/
User interface and dashboard

vit_small_emotion.pth
Trained model weights

requirements.txt
Python dependencies

Dockerfile
Container setup

README.md
Project documentation
Machine Learning Model

Model Architecture
Vision Transformer (ViT)

Base Model
vit-base-patch16-224

Training Dataset
FER2013 Facial Emotion Dataset

Detected Emotions

Happy
Sad
Neutral
Angry
Fear
Disgust
Surprise

Emotion → Stress Mapping

Happy → Low
Neutral → Medium
Sad → Medium
Angry → High
Fear → High
Disgust → High
Surprise → Low

Tech Stack

Frontend
HTML
CSS
JavaScript

Backend
FastAPI
Uvicorn

Machine Learning
PyTorch
Hugging Face Transformers

Computer Vision
OpenCV
MediaPipe
RetinaFace

Deployment Ready
Docker
Render configuration

Installation

Clone the repository

git clone https://github.com/sername/mindcare-ai.git
cd mindcare-ai

Create virtual environment

python -m venv venv
source venv/bin/activate

Install dependencies

pip install -r requirements.txt
Running the Application

Start backend server

uvicorn backend.main:app --reload

Backend will run at

http://127.0.0.1:8000

API documentation

http://127.0.0.1:8000/docs
Inference Modes

Live Mode
Camera based real-time analysis.

Demo Mode
Mock predictions for UI testing.

Manual Mode
User enters mood and stress scale.

Sensor Mode
Analyzes heart rate, HRV, sleep, and self-reported stress.

API Example

Prediction endpoint

POST /predict/image

Response example

{
  "emotion": "happy",
  "stress_level": "low",
  "confidence": 0.87
}
Safety Notice

This system is for research and educational purposes only.

It is not a medical diagnostic tool and should not replace professional mental health care.

Future Improvements

• Mobile application version
• Wearable device integration
• Better emotion datasets
• Real-time stress trend analytics
• Personalized mental wellness recommendations
• Cloud deployment with GPU inference

Author

Chandan Kumar Sah

AI / Machine Learning Project

License

MIT License

Contribution

Contributions are welcome.

Fork the repository

Create a new branch

Submit a pull request

Acknowledgments

FER2013 Dataset
Hugging Face Transformers
OpenCV Community
FastAPI Framework
## Quick Start

### Option 1: Automated Start Script (Recommended)

**For macOS/Linux:**
```bash
./start.sh
```

**For Windows:**
```bash
start.bat
```

This script will:
1. Check and free any ports in use (5500, 8000)
2. Install backend dependencies in a virtual environment
3. Start the backend server (FastAPI) on http://localhost:8000
4. Start the frontend server on http://localhost:5500
5. Open MindCare in your default browser

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
python -m http.server 5500
```

Then open: http://localhost:5500

---

## Problem: "Fail to Load" Error

If you're seeing "fail to load" errors when using the application:

### Cause
The frontend (HTML/JavaScript) tries to communicate with the backend API server at `http://localhost:8000`. If the backend server is not running, all API requests will fail.

### Solution
**You MUST run the backend server alongside the frontend.**

1. **Using the start script (easiest):**
   ```bash
   ./start.sh  # macOS/Linux
   start.bat   # Windows
   ```

2. **Or manually:**
   ```bash
   # Terminal 1 - Start backend
   cd backend
   python main.py
   
   # Terminal 2 - Start frontend
   cd frontend
   python -m http.server 5500
   ```

3. **Or using VS Code Live Server:**
   - The Live Server extension only serves static files (HTML/CSS/JS)
   - It does NOT start the Python backend
   - You still need to run `python backend/main.py` in a separate terminal

---

## Application Features

### Input Methods
- **Camera**: Real-time facial expression analysis
- **Upload**: Upload images for emotion detection
- **Manual**: Self-reported mood and stress assessment
- **Sensor**: Physiological data (heart rate, HRV, sleep, etc.)

### Modes
- **Demo Mode**: Simulated AI responses (works offline)
- **Live AI Mode**: Uses ViT model for real predictions (requires backend)

---

## API Endpoints

When backend is running:
- **API Root**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Image Prediction**: POST /predict/image
- **Manual Prediction**: POST /predict/manual
- **Sensor Prediction**: POST /predict/sensor

---

## Troubleshooting

### Camera Access Issues
- Use HTTPS or http://localhost (not 127.0.0.1)
- Grant camera permissions in browser

### Backend Not Connecting
1. Check if port 8000 is in use: `lsof -i:8000`
2. Kill existing process if needed
3. Restart with: `cd backend && python main.py`

### CORS Errors
The backend is configured with CORS to allow all origins. If you see CORS errors:
- Ensure backend is running on http://localhost:8000
- Check browser console for details

---

## Requirements

- Python 3.8+
- Modern web browser (Chrome, Firefox, Edge, Safari)
- Camera access (for camera feature)

