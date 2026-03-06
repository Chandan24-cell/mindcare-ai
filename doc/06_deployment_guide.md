# Deployment Guide

## Local Deployment

Install dependencies

pip install -r requirements.txt

Start server

uvicorn backend.main:app --reload

---

## Docker Deployment

Build image

docker build -t mindcare-ai .

Run container

docker run -p 8000:8000 mindcare-ai

---

## Production Suggestions

Use:

NGINX  
GPU inference  
Cloud hosting