"""
NeoPredict - Multi-Model Neonatal Health Prediction Pipeline
============================================================
Authors: Research by Alliance University Team
Integration & Inference Engine for NeoPredict Web Platform.

Architecture:
1. Sepsis Classifier: Trained Random Forest model (sepsis_model.pkl.gz)
2. Bradycardia Predictor: RR-dynamics and neonatal threshold model (cardia_dp.ipynb)
3. Apnea of Prematurity Predictor: Respiratory and SpO2 dynamics model (apnea_dp.ipynb)
4. Hypoxia Severity Assessor: SpO2 oxygenation gradient model
5. Contextual Attention Fusion Layer: Dynamic multi-sensor fusion with environmental stressors
"""

import os
import joblib
import numpy as np
import pandas as pd

# Clinical reference values and medians for dataset-2 features
DEFAULT_MEDIANS = {
    'Hour': 19.0,
    'HR': 130.0,            # Neonatal median resting HR
    'O2Sat': 97.0,          # Neonatal median SpO2
    'Temp': 37.0,           # Normal core temperature (C)
    'SBP': 65.0,            # Neonatal systolic BP
    'MAP': 48.0,            # Neonatal mean arterial pressure
    'DBP': 38.0,            # Neonatal diastolic BP
    'Resp': 42.0,           # Neonatal respiration rate (breaths/min)
    'EtCO2': 35.0,
    'BaseExcess': -1.0,
    'HCO3': 24.0,
    'FiO2': 0.35,
    'pH': 7.38,
    'PaCO2': 39.5,
    'SaO2': 97.0,
    'AST': 127.0,
    'BUN': 18.0,
    'Alkalinephos': 87.5,
    'Calcium': 8.4,
    'Chloride': 106.0,
    'Creatinine': 0.6,
    'Bilirubin_direct': 1.5,
    'Glucose': 95.0,
    'Lactate': 2.0,
    'Magnesium': 2.0,
    'Phosphate': 3.3,
    'Potassium': 4.1,
    'Bilirubin_total': 1.2,
    'TroponinI': 1.6,
    'Hct': 35.0,
    'Hgb': 12.0,
    'PTT': 32.0,
    'WBC': 11.5,
    'Fibrinogen': 280.0,
    'Platelets': 210.0,
    'Age': 0.05,            # Newborn age in days/weeks
    'Gender': 1.0,
    'Unit1': 1.0,
    'Unit2': 0.0,
    'HospAdmTime': -2.43,
    'ICULOS': 21.0,
    'Patient_ID': 1001.0,
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sepsis_model.pkl.gz")


class NeoPredictPipeline:
    def __init__(self):
        self.sepsis_model = None
        self.feature_names = None
        self._load_sepsis_model()

    def _load_sepsis_model(self):
        """Loads pre-trained compressed sepsis model if present."""
        if os.path.exists(MODEL_PATH):
            try:
                self.sepsis_model = joblib.load(MODEL_PATH)
                if hasattr(self.sepsis_model, "feature_names_in_"):
                    self.feature_names = list(self.sepsis_model.feature_names_in_)
                print(f"[NeoPredict] Sepsis model loaded successfully from {MODEL_PATH}")
            except Exception as e:
                print(f"[NeoPredict] Warning loading sepsis model: {e}")
                self.sepsis_model = None
        else:
            print(f"[NeoPredict] Warning: {MODEL_PATH} not found. Running with rule-based fallback.")

    def compute_bradycardia_risk(self, hr: float) -> dict:
        """
        Calculates Neonatal Bradycardia Risk based on PICSDB findings (cardia_dp.ipynb).
        Normal neonatal heart rate: 110 - 160 bpm.
        Bradycardia: HR < 100 bpm.
        Severe Bradycardia: HR < 80 bpm.
        """
        if hr >= 120:
            risk = max(2.0, (140 - hr) * 0.1)
        elif 100 <= hr < 120:
            risk = 10.0 + (120 - hr) * 1.5
        elif 80 <= hr < 100:
            risk = 40.0 + (100 - hr) * 2.2
        else:  # hr < 80 bpm (critical)
            risk = min(99.5, 84.0 + (80 - hr) * 0.75)

        risk = float(np.clip(risk, 1.0, 99.0))
        level = "Normal" if risk < 30 else ("Moderate" if risk < 65 else "Critical")
        return {
            "risk_percent": round(risk, 1),
            "level": level,
            "condition": "Bradycardia",
            "indicator": f"{hr:.0f} bpm (Threshold < 100 bpm)"
        }

    def compute_apnea_risk(self, resp: float, spo2: float, hr: float) -> dict:
        """
        Calculates Apnea of Prematurity (AOP) risk based on apnea_dp.ipynb findings.
        Normal RR: 30 - 60 breaths/min.
        Bradypnea / Respiratory pause: RR < 20 breaths/min.
        Severe Apnea: RR < 15 breaths/min accompanied by desaturation and cardiac slowing.
        """
        if resp >= 35:
            r_score = max(3.0, (40 - resp) * 0.2)
        elif 25 <= resp < 35:
            r_score = 15.0 + (35 - resp) * 2.5
        elif 18 <= resp < 25:
            r_score = 40.0 + (25 - resp) * 4.0
        else:  # resp < 18
            r_score = min(95.0, 68.0 + (18 - resp) * 3.5)

        # SpO2 coupling (apnea causes rapid desaturation)
        if spo2 < 90:
            spo2_penalty = (90 - spo2) * 2.8
        else:
            spo2_penalty = 0.0

        # Bradycardia coupling (reflex bradycardia during prolonged apnea)
        hr_penalty = (100 - hr) * 0.5 if hr < 100 else 0.0

        total_risk = float(np.clip(r_score + spo2_penalty + hr_penalty, 2.0, 99.0))
        level = "Normal" if total_risk < 30 else ("Moderate" if total_risk < 65 else "Critical")
        return {
            "risk_percent": round(total_risk, 1),
            "level": level,
            "condition": "Apnea of Prematurity",
            "indicator": f"{resp:.0f} breaths/min (Threshold < 20)"
        }

    def compute_hypoxia_risk(self, spo2: float, resp: float) -> dict:
        """
        Calculates Hypoxia / Desaturation Severity Risk.
        Normal SpO2: 95 - 100%.
        Mild: 90 - 94%.
        Moderate: 85 - 89%.
        Severe: < 85%.
        """
        if spo2 >= 95:
            risk = max(2.0, (100 - spo2) * 1.5)
        elif 90 <= spo2 < 95:
            risk = 15.0 + (95 - spo2) * 6.0
        elif 85 <= spo2 < 90:
            risk = 45.0 + (90 - spo2) * 6.5
        else:  # spo2 < 85
            risk = min(99.0, 78.0 + (85 - spo2) * 2.2)

        if resp < 20 or resp > 60:
            risk += 6.0

        risk = float(np.clip(risk, 1.0, 99.0))
        level = "Normal" if risk < 30 else ("Moderate" if risk < 65 else "Critical")
        return {
            "risk_percent": round(risk, 1),
            "level": level,
            "condition": "Hypoxia / Desaturation",
            "indicator": f"{spo2:.1f}% SpO2 (Target >= 95%)"
        }

    def compute_sepsis_risk(self, hr: float, temp: float, resp: float, spo2: float,
                            sbp: float, dbp: float) -> dict:
        """
        Calculates Neonatal Sepsis Risk using the trained Random Forest model (sepsis_dp.ipynb).
        """
        map_bp = round(dbp + (sbp - dbp) / 3.0, 1)

        # If model is loaded, run feature inference
        model_risk = None
        if self.sepsis_model is not None and self.feature_names:
            try:
                row = dict(DEFAULT_MEDIANS)
                row["HR"] = float(hr)
                row["Temp"] = float(temp)
                row["Resp"] = float(resp)
                row["O2Sat"] = float(spo2)
                row["SBP"] = float(sbp)
                row["DBP"] = float(dbp)
                row["MAP"] = float(map_bp)

                input_df = pd.DataFrame([row])[self.feature_names]
                probs = self.sepsis_model.predict_proba(input_df)[0]
                model_risk = float(probs[1] * 100.0)
            except Exception as e:
                print(f"[NeoPredict] Inference exception: {e}")
                model_risk = None

        # Clinical neonatal SIRS criteria calibration
        sirs_score = 0
        if temp < 36.5 or temp > 38.0:
            sirs_score += 30
        if hr > 175 or hr < 100:
            sirs_score += 25
        if resp > 60 or resp < 25:
            sirs_score += 20
        if map_bp < 40:
            sirs_score += 15
        if spo2 < 92:
            sirs_score += 10

        sirs_risk = float(np.clip(sirs_score, 4.0, 96.0))

        if model_risk is not None:
            final_risk = 0.65 * model_risk + 0.35 * sirs_risk
        else:
            final_risk = sirs_risk

        final_risk = float(np.clip(final_risk, 2.0, 98.0))
        level = "Normal" if final_risk < 30 else ("Moderate" if final_risk < 60 else "Critical")
        return {
            "risk_percent": round(final_risk, 1),
            "level": level,
            "condition": "Neonatal Sepsis",
            "indicator": f"Temp: {temp:.1f}C, HR: {hr:.0f}, Resp: {resp:.0f}"
        }

    def compute_environmental_stress(self, room_temp: float, humidity: float, aqi: float) -> dict:
        """
        Evaluates ambient NICU/incubator environment.
        Ideal incubator/room temp: 24 - 28 C.
        Ideal humidity: 45 - 65%.
        Healthy AQI: < 50.
        """
        stress = 0.0
        if room_temp < 22:
            stress += (22 - room_temp) * 6.0  # Hypothermia risk
        elif room_temp > 30:
            stress += (room_temp - 30) * 5.0  # Hyperthermia risk

        if humidity < 35:
            stress += (35 - humidity) * 0.8   # Transepidermal water loss
        elif humidity > 75:
            stress += (humidity - 75) * 0.7   # Microbial risk

        if aqi > 50:
            stress += (aqi - 50) * 0.45

        stress = float(np.clip(stress, 0.0, 100.0))
        return {
            "stress_index": round(stress, 1),
            "room_temp": room_temp,
            "humidity": humidity,
            "aqi": aqi
        }

    def attention_fusion(self, risks: dict, env_stress: float) -> dict:
        """
        Multi-Modal Attention-Based Fusion Layer (Paper Section 3.3).
        Computes dynamic attention weights for each subsystem and aggregates
        a unified NeoPredict Health Risk Index.
        """
        keys = ["Apnea", "Bradycardia", "Sepsis", "Hypoxia"]
        values = np.array([
            risks["apnea"]["risk_percent"],
            risks["bradycardia"]["risk_percent"],
            risks["sepsis"]["risk_percent"],
            risks["hypoxia"]["risk_percent"]
        ], dtype=float)

        env_modulation = 1.0 + (env_stress / 100.0) * 0.35

        tau = 28.0
        exp_vals = np.exp((values * env_modulation) / tau)
        attention_weights = exp_vals / np.sum(exp_vals)

        composite_score = float(np.sum(attention_weights * values))
        if env_stress > 40:
            composite_score = min(99.0, composite_score + (env_stress - 40) * 0.15)

        composite_score = float(np.clip(composite_score, 1.0, 99.0))

        if composite_score < 25:
            status = "NORMAL"
            badge_class = "success"
            action = "Patient vitals are stable within normal physiological boundaries. Continue standard non-invasive monitoring."
        elif composite_score < 50:
            status = "MODERATE RISK"
            badge_class = "warning"
            action = "Mild physiological or environmental anomaly detected. Verify sensor attachment, monitor respiratory trends, and adjust incubator climate."
        elif composite_score < 75:
            status = "HIGH RISK"
            badge_class = "danger"
            action = "Elevated risk profile detected across key vitals. Alert bedside nursing staff for clinical assessment, check airway patency, and evaluate blood gas."
        else:
            status = "CRITICAL ALERT"
            badge_class = "dark-danger"
            action = "CRITICAL ALERT: Imminent risk of acute decompensation. Immediate NICU physician intervention, oxygenation support, and emergency protocol activation required!"

        weights_dict = {
            keys[i]: round(float(attention_weights[i]) * 100, 1) for i in range(len(keys))
        }

        return {
            "composite_score": round(composite_score, 1),
            "status": status,
            "badge_class": badge_class,
            "attention_weights": weights_dict,
            "action_plan": action
        }

    def predict(self, data: dict) -> dict:
        """
        Full end-to-end multi-modal inference pipeline.
        """
        hr = float(data.get("hr", 130.0))
        resp = float(data.get("resp", 42.0))
        spo2 = float(data.get("spo2", 97.0))
        temp = float(data.get("temp", 37.0))
        sbp = float(data.get("sbp", 65.0))
        dbp = float(data.get("dbp", 38.0))

        room_temp = float(data.get("room_temp", 26.0))
        humidity = float(data.get("humidity", 55.0))
        aqi = float(data.get("aqi", 35.0))

        brady = self.compute_bradycardia_risk(hr)
        apnea = self.compute_apnea_risk(resp, spo2, hr)
        hypoxia = self.compute_hypoxia_risk(spo2, resp)
        sepsis = self.compute_sepsis_risk(hr, temp, resp, spo2, sbp, dbp)
        env = self.compute_environmental_stress(room_temp, humidity, aqi)

        condition_risks = {
            "bradycardia": brady,
            "apnea": apnea,
            "hypoxia": hypoxia,
            "sepsis": sepsis
        }

        fusion_result = self.attention_fusion(condition_risks, env["stress_index"])

        return {
            "success": True,
            "inputs": {
                "physiological": {
                    "heart_rate": hr,
                    "respiration_rate": resp,
                    "spo2": spo2,
                    "body_temperature": temp,
                    "blood_pressure": f"{sbp:.0f}/{dbp:.0f} mmHg"
                },
                "environmental": {
                    "room_temperature": room_temp,
                    "humidity": humidity,
                    "aqi": aqi
                }
            },
            "condition_risks": condition_risks,
            "environmental_stress": env,
            "fusion": fusion_result
        }


# Singleton pipeline instance
pipeline = NeoPredictPipeline()
