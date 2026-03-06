# API Documentation

Base URL

http://localhost:8000

---

## Image Prediction

POST /predict/image

Parameters

file : image file  
mode : real or demo

Response

{
  "emotion": "happy",
  "stress_level": "low",
  "confidence": 0.84
}

---

## Manual Input

POST /predict/manual

Body

{
  "mood": "sad",
  "stress_scale": 7
}

---

## Sensor Input

POST /predict/sensor

Body

{
  "heart_rate": 92,
  "hrv": 45,
  "sleep_hours": 5,
  "self_mood": "tired",
  "stress_scale": 6
}