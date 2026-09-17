# NeoPredict — Neonatal Health Prediction System

[![Render Deployment](https://img.shields.io/badge/Deploy%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://neopredict.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.7%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

A deep learning-based multi-modal web platform for early neonatal health assessment and real-time risk prediction in Neonatal Intensive Care Units (NICUs).

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

## Architecture

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

## License & Ethical Disclosure
This software is intended for research, educational, and clinical decision support purposes. Clinical decisions must always be made by licensed healthcare professionals and NICU pediatricians.
