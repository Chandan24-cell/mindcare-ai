# System Architecture

## Overview

MindCare AI is a multi-modal stress analysis platform combining
computer vision, behavioral input, and physiological signals.

The system follows a modular architecture separating:

• User Interface
• API Layer
• Machine Learning Inference
• Model Storage
• Notification System

---

## High Level Architecture

User
   │
   ▼
Frontend Dashboard
   │
   ▼
FastAPI Backend
   │
   ├── Image Processing
   ├── Sensor Analysis
   ├── Manual Input Analysis
   │
   ▼
ML Inference Engine
   │
   ▼
Emotion + Stress Prediction
   │
   ▼
Suggestion Engine
   │
   ▼
Dashboard Response

---

## Core Components

### Frontend

Responsible for:

• Camera capture
• Image upload
• Displaying predictions
• Showing notifications
• Dashboard interaction

Technologies:
HTML, CSS, JavaScript

---

### Backend API

Handles:

• Request validation
• Routing
• Inference execution
• Response formatting

Technology:
FastAPI

---

### ML Engine

Responsible for:

• Face detection
• Image preprocessing
• Emotion prediction
• Stress level mapping

Technologies:

PyTorch  
Vision Transformer  
OpenCV  
MediaPipe  
RetinaFace