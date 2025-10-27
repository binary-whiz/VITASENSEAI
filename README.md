# VitasenseAI — Deployment Guide

This repository runs a small Flask app that captures PPG frames and returns vitals and an AI assessment via the Google Generative API.

This README explains quick ways to host the app publicly.

Important: keep your API key secret. Do NOT commit `.env` to git.

## Recommended: Container deploy (Google Cloud Run / Render / Railway)

1. Build image locally (optional):

```powershell
docker build -t vitasenseai:latest .
docker run -p 8080:8080 --env-file .env vitasenseai:latest
```

2. Deploy to Google Cloud Run (example):

- Install and authenticate gcloud, enable Cloud Run API and Container Registry/Artifact Registry.
- Build and push image, then deploy:

```powershell
gcloud builds submit --tag gcr.io/PROJECT-ID/vitasenseai
gcloud run deploy vitasenseai --image gcr.io/PROJECT-ID/vitasenseai --platform managed --region us-central1 --allow-unauthenticated --set-env-vars GOOGLE_API_KEY="YOUR_KEY"
```

3. Deploy to Render (simple web service):

- Create a new Web Service on Render, connect your Git repo.
- Use the Dockerfile option (the repo contains a `Dockerfile`), set the port to `8080` and add an environment variable `GOOGLE_API_KEY` in Render's dashboard.

## Quick deploy to Heroku (not recommended for heavy CPU / CV work)

Heroku's free tier is no longer recommended — prefer Cloud Run or Render.

## Local notes
- Requirements are in `requirements.txt`. On Windows, building packages like `scipy` and `opencv-python` may require appropriate wheels — the Dockerfile uses Debian slim and installs system deps.
- The app reads `GOOGLE_API_KEY` from the environment (or `.env`). Use `python .\app.py` for local debugging.

## Security
- Put your API key in environment variables on the host platform (Render/Cloud Run/Heroku dashboard). Do not commit keys.

## Troubleshooting
- If you hit quota limits for the Generative API, enable billing or switch to a smaller model in `.env` (change `GEMINI_MODEL`).
