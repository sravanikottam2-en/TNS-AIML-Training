# Diabetes Risk Prediction (Supervised Learning)

A Flask web application demonstrating registration/login, a supervised ML prediction workflow, risk percentage display, dashboard, prediction history, general health tips, and downloadable reports.

## Important medical limitation
**This is an educational software demo, not a medical device.** The included classifier is trained on generated synthetic data with simulated labels, not real patient records or a clinically validated dataset. Its percentage is only the model's output and must not be interpreted as an individual's true probability of diabetes. Do not use it for diagnosis, treatment, or medical decisions. For real screening, consult a qualified healthcare professional.

## Features
- User registration and login (passwords hashed with Werkzeug)
- Supervised Random Forest classification with `predict_proba`
- Risk percentage and Low / Moderate / High demo categories
- Personal dashboard and latest predictions
- SQLite prediction history
- General wellness tips
- Downloadable TXT report for each saved prediction

## Requirements
- Python 3.10 or newer
- pip

## Run locally (Windows)
1. Extract the ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal and create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   python app.py
   ```
6. Open http://127.0.0.1:5000

On first run, the model is trained automatically and saved as `diabetes_model.joblib`. SQLite database `diabetes_app.db` is created automatically.

## Deploy (Render / similar)
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Set environment variable `SECRET_KEY` to a long random secret.
- Use persistent storage for `diabetes_app.db` and `diabetes_model.joblib`; ephemeral deployment filesystems may reset on redeploy.
- For production, configure HTTPS, CSRF protection, rate limiting, secure cookie settings, backups, and managed persistent database storage.

## Input fields
Pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age. These fields resemble commonly used diabetes dataset columns; this demo does not use an actual clinical dataset.
