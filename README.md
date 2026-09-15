# NeoPredict — Neonatal Health Prediction System

[![Render Deployment](https://img.shields.io/badge/Deploy%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://neopredict.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.7%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

A deep learning-based multi-modal web platform for early neonatal health assessment and real-time risk prediction in Neonatal Intensive Care Units (NICUs).

---

## 📑 Research Background & Abstract

> **Research Paper**: *Deep Learning-Based Multi-Model Neonatal Health Prediction System with Real-Time Risk Assessment (NeoPredict)*  
> **Authors**: Gundupalli Tejaswi Reddy, Choda Lahari, D Poojitha, Shaik Anum Tabasum  
> **Institution**: Alliance School of Advanced Computing, Alliance University  

Traditional neonatal monitoring systems often depend on predefined threshold rules, which generate frequent false alarms (85-90% false positive rate) and provide limited predictive insight. **NeoPredict** introduces an attention-based multi-model framework that continuously integrates physiological vital signs (Heart Rate, Respiration Rate, SpO₂, Temperature, Blood Pressure) with ambient environmental parameters (Incubator Temperature, Humidity, Air Quality Index) to predict acute conditions:
- **Apnea of Prematurity (AOP)**
- **Neonatal Bradycardia**
- **Neonatal Sepsis**
- **Hypoxia / Desaturation**

Validated on PhysioNet PICSDB, Neonatal Sepsis (Figshare), and PhysioNet Apnea-ECG datasets, achieving:
- **96.2% Accuracy**
- **95.4% Precision**
- **96.7% Recall**
- **96.0% F1-Score**

---

## 🏗️ Architecture

```
                               ┌────────────────────────────────┐
                               │   Physiological Sensor Stream  │
                               │   (HR, RR, SpO2, Temp, BP)     │
                               └──────────────┬─────────────────┘
                                              │
                   ┌──────────────────────────┼──────────────────────────┐
                   ▼                          ▼                          ▼
        ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
        │  Bradycardia Model  │    │     Apnea Model     │    │    Sepsis Model     │
        │   (PICSDB / RR)     │    │   (Apnea-ECG / RR)  │    │  (Random Forest 30) │
        └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘
                   │                          │                          │
                   └──────────────────┐       │       ┌──────────────────┘
                                      ▼       ▼       ▼
                               ┌────────────────────────────────┐
                               │    Contextual Attention Layer   │ ◄── Ambient Environment
                               │  Softmax Sensor Salience (τ)   │     (Room Temp, Hum, AQI)
                               └──────────────┬─────────────────┘
                                              ▼
                               ┌────────────────────────────────┐
                               │ Composite NeoPredict Risk Index│
                               │ (Normal / Moderate / Critical) │
                               └────────────────────────────────┘
```

---

## 💻 Local Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/neopredict.git
cd neopredict
```

### 2. Create virtual environment & install requirements
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the development server
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

Or run with Gunicorn (production WSGI):
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

---

## 🚀 Deployment on Render

### Step 1: Push Project to GitHub
```bash
git init
git add .
git commit -m "Initial commit: NeoPredict production web application"
git branch -M main
git remote add origin https://github.com/<your-username>/neopredict.git
git push -u origin main
```

### Step 2: Deploy as a Web Service on Render
1. Log in to [dashboard.render.com](https://dashboard.render.com/).
2. Click **New +** &rarr; **Web Service**.
3. Connect your GitHub repository (`neopredict`).
4. Fill in the configuration:
   - **Name**: `neopredict` *(Ensures URL is `neopredict.onrender.com`)*
   - **Region**: Choose the closest region (e.g., Oregon, Frankfurt, Singapore).
   - **Branch**: `main`
   - **Root Directory**: *(Leave blank)*
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --workers=2 --threads=2 --timeout=120 app:app`
   - **Instance Type**: `Free`
5. Click **Create Web Service**.
6. Within 2-3 minutes, your live web application will be accessible at:
   **`https://neopredict.onrender.com`**

---

## 📡 REST API Documentation

### Assess Patient Risks: `POST /api/predict`
Request:
```bash
curl -X POST https://neopredict.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hr": 135,
    "resp": 42,
    "spo2": 98.0,
    "temp": 37.0,
    "sbp": 65,
    "dbp": 38,
    "room_temp": 26.0,
    "humidity": 55,
    "aqi": 35
  }'
```

Response:
```json
{
  "success": true,
  "condition_risks": {
    "apnea": {
      "condition": "Apnea of Prematurity",
      "indicator": "42 breaths/min (Threshold < 20)",
      "level": "Normal",
      "risk_percent": 3.0
    },
    "bradycardia": {
      "condition": "Bradycardia",
      "indicator": "135 bpm (Threshold < 100 bpm)",
      "level": "Normal",
      "risk_percent": 2.0
    },
    "hypoxia": {
      "condition": "Hypoxia / Desaturation",
      "indicator": "98.0% SpO2 (Target >= 95%)",
      "level": "Normal",
      "risk_percent": 3.0
    },
    "sepsis": {
      "condition": "Neonatal Sepsis",
      "indicator": "Temp: 37.0C, HR: 135, Resp: 42",
      "level": "Normal",
      "risk_percent": 20.9
    }
  },
  "environmental_stress": {
    "aqi": 35.0,
    "humidity": 55.0,
    "room_temp": 26.0,
    "stress_index": 0.0
  },
  "fusion": {
    "action_plan": "Patient vitals are stable within normal physiological boundaries. Continue standard non-invasive monitoring.",
    "attention_weights": {
      "Apnea": 20.6,
      "Bradycardia": 19.9,
      "Hypoxia": 20.6,
      "Sepsis": 39.0
    },
    "badge_class": "success",
    "composite_score": 9.8,
    "status": "NORMAL"
  }
}
```

### Health Check: `GET /api/health`
```json
{
  "model_loaded": true,
  "service": "NeoPredict Neonatal Health Prediction System",
  "status": "healthy",
  "version": "1.0.0"
}
```

---

## ⚖️ License & Ethical Disclosure
This software is intended for research, educational, and clinical decision support purposes. Clinical decisions must always be made by licensed healthcare professionals and NICU pediatricians.
