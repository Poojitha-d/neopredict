"""
NeoPredict - Neonatal Health Prediction System
==============================================
Flask Web Service & REST API for Render Deployment.
"""

import os
from flask import Flask, render_template, request, jsonify
from model_pipeline import pipeline

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Renders the NeoPredict Clinical Assessment & Monitoring Dashboard."""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for clinical assessment.
    Accepts JSON body or Form data.
    """
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        if not data:
            return jsonify({"success": False, "error": "No input data provided"}), 400

        result = pipeline.predict(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for Render monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "NeoPredict Neonatal Health Prediction System",
        "model_loaded": pipeline.sepsis_model is not None,
        "version": "1.0.0"
    }), 200

if __name__ == "__main__":
    # Render passes the port in the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    # In production on Render, Gunicorn will be used instead of app.run
    app.run(host="0.0.0.0", port=port, debug=False)
