/**
 * NeoPredict Interactive Web Controller
 */

const SCENARIOS = {
    normal: {
        hr: 135,
        resp: 42,
        spo2: 98.0,
        temp: 36.8,
        sbp: 65,
        dbp: 38,
        room_temp: 26.0,
        humidity: 55,
        aqi: 30
    },
    apnea: {
        hr: 110,
        resp: 14,
        spo2: 84.0,
        temp: 36.6,
        sbp: 58,
        dbp: 34,
        room_temp: 25.5,
        humidity: 50,
        aqi: 40
    },
    brady: {
        hr: 78,
        resp: 38,
        spo2: 91.0,
        temp: 36.5,
        sbp: 52,
        dbp: 30,
        room_temp: 24.0,
        humidity: 50,
        aqi: 45
    },
    sepsis: {
        hr: 188,
        resp: 68,
        spo2: 91.5,
        temp: 38.7,
        sbp: 48,
        dbp: 26,
        room_temp: 27.0,
        humidity: 58,
        aqi: 60
    },
    critical: {
        hr: 75,
        resp: 12,
        spo2: 78.0,
        temp: 35.6,
        sbp: 44,
        dbp: 22,
        room_temp: 20.0,
        humidity: 30,
        aqi: 98
    }
};

function loadScenario(type) {
    const s = SCENARIOS[type];
    if (!s) return;

    document.getElementById("hr").value = s.hr;
    document.getElementById("resp").value = s.resp;
    document.getElementById("spo2").value = s.spo2;
    document.getElementById("temp").value = s.temp;
    document.getElementById("sbp").value = s.sbp;
    document.getElementById("dbp").value = s.dbp;
    document.getElementById("room_temp").value = s.room_temp;
    document.getElementById("humidity").value = s.humidity;
    document.getElementById("aqi").value = s.aqi;

    // Automatically trigger assessment
    document.getElementById("predictionForm").dispatchEvent(new Event("submit"));
}

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const submitBtn = document.getElementById("submitBtn");
    const spinner = document.getElementById("loadingSpinner");
    const btnIcon = document.getElementById("btnIcon");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // UI loading state
        submitBtn.disabled = true;
        spinner.classList.remove("d-none");
        btnIcon.classList.add("d-none");

        const formData = new FormData(form);
        const payload = {};
        formData.forEach((val, key) => {
            payload[key] = parseFloat(val);
        });

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (result.success) {
                renderResults(result);
            } else {
                alert("Assessment error: " + (result.error || "Unknown error"));
            }
        } catch (err) {
            console.error("Network error:", err);
            alert("Error communicating with NeoPredict inference engine.");
        } finally {
            submitBtn.disabled = false;
            spinner.classList.add("d-none");
            btnIcon.classList.remove("d-none");
        }
    });

    // Run initial assessment with default normal values
    loadScenario("normal");
});

function renderResults(data) {
    const fusion = data.fusion;
    const risks = data.condition_risks;
    const env = data.environmental_stress;

    // Timestamp
    const now = new Date();
    document.getElementById("timestampBadge").textContent = "Evaluated at " + now.toLocaleTimeString();

    // 1. Composite Score Box
    const compBox = document.getElementById("compositeBox");
    const compScoreElem = document.getElementById("compositeScore");
    const statusPill = document.getElementById("statusPill");
    const actionPlan = document.getElementById("actionPlan");

    compScoreElem.textContent = fusion.composite_score;
    statusPill.textContent = fusion.status;
    actionPlan.textContent = fusion.action_plan;

    // Remove old classes
    compBox.className = "composite-panel p-4 rounded-4 text-center mb-4 transition-all";
    statusPill.className = "badge status-pill px-3 py-2 fs-6 rounded-pill";

    if (fusion.status === "NORMAL") {
        compBox.classList.add("state-normal");
        statusPill.classList.add("bg-success");
    } else if (fusion.status === "MODERATE RISK") {
        compBox.classList.add("state-warning");
        statusPill.classList.add("bg-warning");
    } else if (fusion.status === "HIGH RISK") {
        compBox.classList.add("state-danger");
        statusPill.classList.add("bg-danger");
    } else {
        compBox.classList.add("state-critical");
        statusPill.classList.add("bg-dark-danger");
    }

    // 2. Condition Cards
    updateConditionCard("apnea", risks.apnea);
    updateConditionCard("brady", risks.bradycardia);
    updateConditionCard("sepsis", risks.sepsis);
    updateConditionCard("hypoxia", risks.hypoxia);

    // 3. Environmental Stress & Attention Weights
    document.getElementById("envStressBadge").textContent = "Env Stress: " + env.stress_index + "%";

    const weights = fusion.attention_weights;
    document.getElementById("attApnea").style.width = weights.Apnea + "%";
    document.getElementById("attApnea").textContent = "Apnea " + weights.Apnea + "%";

    document.getElementById("attBrady").style.width = weights.Bradycardia + "%";
    document.getElementById("attBrady").textContent = "Brady " + weights.Bradycardia + "%";

    document.getElementById("attSepsis").style.width = weights.Sepsis + "%";
    document.getElementById("attSepsis").textContent = "Sepsis " + weights.Sepsis + "%";

    document.getElementById("attHypoxia").style.width = weights.Hypoxia + "%";
    document.getElementById("attHypoxia").textContent = "Hypoxia " + weights.Hypoxia + "%";

    document.getElementById("attentionWeightLabels").innerHTML = `
        <span>Apnea: <strong>${weights.Apnea}%</strong></span>
        <span>Brady: <strong>${weights.Bradycardia}%</strong></span>
        <span>Sepsis: <strong>${weights.Sepsis}%</strong></span>
        <span>Hypoxia: <strong>${weights.Hypoxia}%</strong></span>
    `;
}

function updateConditionCard(prefix, cond) {
    const scoreElem = document.getElementById(prefix + "Score");
    const barElem = document.getElementById(prefix + "Bar");
    const levelElem = document.getElementById(prefix + "Level");
    const indElem = document.getElementById(prefix + "Ind");

    scoreElem.textContent = cond.risk_percent + "%";
    barElem.style.width = cond.risk_percent + "%";
    levelElem.textContent = cond.level;
    indElem.textContent = cond.indicator;

    // Badge styling
    levelElem.className = "badge";
    if (cond.level === "Normal") {
        levelElem.classList.add("bg-success-subtle", "text-success");
    } else if (cond.level === "Moderate") {
        levelElem.classList.add("bg-warning-subtle", "text-warning");
    } else {
        levelElem.classList.add("bg-danger-subtle", "text-danger");
    }
}
