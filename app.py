from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import math
import urllib.parse
import urllib.request

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "ml_system.joblib"
METRICS_PATH = ROOT / "model" / "model_metrics.json"

if not MODEL_PATH.exists():
    from train_model import main as train_model
    train_model()

BUNDLE = joblib.load(MODEL_PATH)
VISIBILITY_MODEL = BUNDLE["visibility_model"]
RISK_MODEL = BUNDLE["risk_model"]
SAFETY_MODEL = BUNDLE["safety_model"]
DEPARTURE_MODEL = BUNDLE["departure_model"]
FEATURES = BUNDLE["features"]
AUX_FEATURES = BUNDLE["aux_features"]
app = Flask(__name__)

KNOWN_LOCATIONS = {
    "prayagraj": (25.4358, 81.8463, "Prayagraj, Uttar Pradesh, India"),
    "allahabad": (25.4358, 81.8463, "Prayagraj, Uttar Pradesh, India"),
    "varanasi": (25.3176, 82.9739, "Varanasi, Uttar Pradesh, India"),
    "jhusi": (25.4380, 81.9024, "Jhusi, Prayagraj, Uttar Pradesh, India"),
    "katra": (25.4680, 81.8600, "Old Katra, Prayagraj, Uttar Pradesh, India"),
    "lucknow": (26.8467, 80.9462, "Lucknow, Uttar Pradesh, India"),
    "kanpur": (26.4499, 80.3319, "Kanpur, Uttar Pradesh, India"),
    "ayodhya": (26.7922, 82.1998, "Ayodhya, Uttar Pradesh, India"),
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _frame(data: dict, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: float(data[column]) for column in columns}], columns=columns)


def _probability_confidence(model, frame: pd.DataFrame) -> float:
    if not hasattr(model, "predict_proba"):
        return 0.75
    probs = model.predict_proba(frame)[0]
    return float(np.max(probs))


def _visibility_confidence(row: dict[str, float], prediction: float) -> float:
    """ML-derived uncertainty using Random Forest tree dispersion or calibrated error."""
    model = VISIBILITY_MODEL.named_steps.get("model")
    transformed = row_frame = _frame(row, FEATURES)
    imputer = VISIBILITY_MODEL.named_steps.get("imputer")
    if imputer is not None:
        transformed = imputer.transform(row_frame)
    scaler = VISIBILITY_MODEL.named_steps.get("scaler")
    if scaler is not None:
        transformed = scaler.transform(transformed)

    if hasattr(model, "estimators_") and not isinstance(model.estimators_, np.ndarray):
        tree_predictions = np.array([tree.predict(transformed)[0] for tree in model.estimators_])
        spread = float(np.std(tree_predictions))
    else:
        q50 = max(float(BUNDLE.get("confidence_error_q50", 0.7)), 0.05)
        q90 = max(float(BUNDLE.get("confidence_error_q90", 1.8)), q50)
        # Confidence decreases near difficult low-visibility conditions.
        spread = q50 + max(0, 6 - prediction) / 6 * (q90 - q50)
    return clamp(1.0 - spread / max(prediction + 2.0, 4.0), 0.55, 0.97)


def predict_one(data: dict) -> dict:
    row = {feature: float(data[feature]) for feature in FEATURES}
    base_frame = _frame(row, FEATURES)
    visibility = float(VISIBILITY_MODEL.predict(base_frame)[0])
    visibility = round(clamp(visibility, 0.1, 25), 2)

    augmented = {**row, "predicted_visibility_km": visibility}
    aux_frame = _frame(augmented, AUX_FEATURES)
    risk = str(RISK_MODEL.predict(aux_frame)[0])
    safety = int(round(clamp(float(SAFETY_MODEL.predict(aux_frame)[0]), 0, 100)))
    departure = str(DEPARTURE_MODEL.predict(aux_frame)[0])

    risk_conf = _probability_confidence(RISK_MODEL, aux_frame)
    departure_conf = _probability_confidence(DEPARTURE_MODEL, aux_frame)
    vis_conf = _visibility_confidence(row, visibility)
    confidence = int(round(clamp((risk_conf * 0.35 + departure_conf * 0.25 + vis_conf * 0.40) * 100, 55, 97)))

    return {
        "predicted_visibility_km": visibility,
        "risk_level": risk,
        "safety_score": safety,
        "confidence_percent": confidence,
        "departure_action": departure,
        "models": {
            "visibility": BUNDLE["visibility_model_name"],
            "risk": "Random Forest Classifier",
            "safety": "Gradient Boosting Regressor",
            "departure": "Gradient Boosting Classifier",
        },
    }


def fetch_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "VisibilityPredictionModel/3.0 educational-project"})
    with urllib.request.urlopen(req, timeout=12) as response:
        return json.load(response)


def geocode(text: str) -> dict:
    cleaned = text.strip().lower()
    for key, (lat, lon, name) in KNOWN_LOCATIONS.items():
        if cleaned == key or cleaned.startswith(key + ","):
            return {"name": name, "lat": lat, "lon": lon, "source": "local-fallback"}
    query = urllib.parse.urlencode({"q": text, "format": "jsonv2", "limit": 1, "countrycodes": "in"})
    data = fetch_json("https://nominatim.openstreetmap.org/search?" + query)
    if not data:
        raise ValueError(f"Location not found: {text}")
    return {
        "name": data[0]["display_name"], "lat": float(data[0]["lat"]),
        "lon": float(data[0]["lon"]), "source": "OpenStreetMap Nominatim",
    }


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def estimated_environment(lat: float, lon: float, index: int, departure: datetime) -> dict:
    hour = departure.hour
    daylight = max(0, math.sin((hour - 6) / 12 * math.pi))
    temp = 21 + 10 * daylight + (lat % 1) * 2
    humidity = clamp(86 - 28 * daylight + (index % 2) * 4, 35, 96)
    rain = max(0, (humidity - 72) * 0.045)
    return {
        "temperature": round(temp, 1), "humidity": round(humidity, 1),
        "pressure": round(1008 + math.sin(lon) * 4, 1),
        "wind_speed": round(5 + index * 1.4, 1), "precipitation": round(rain, 2),
        "cloud_cover": round(clamp(humidity - 20, 10, 98), 1),
        "aqi": round(68 + index * 6), "hour": hour, "month": departure.month,
        "data_source": "offline demonstration fallback",
    }


def weather(lat: float, lon: float, index: int, departure: datetime) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation,cloud_cover",
        "timezone": "auto",
    }
    try:
        current = fetch_json("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params))["current"]
        return {
            "temperature": current["temperature_2m"], "humidity": current["relative_humidity_2m"],
            "pressure": current["surface_pressure"], "wind_speed": current["wind_speed_10m"],
            "precipitation": current["precipitation"], "cloud_cover": current["cloud_cover"],
            "aqi": 75 + index * 4, "hour": departure.hour, "month": departure.month,
            "data_source": "Open-Meteo live weather",
        }
    except Exception:
        return estimated_environment(lat, lon, index, departure)


def reverse_geocode(lat: float, lon: float, fallback: str) -> str:
    """Return a short, human-readable locality name for a route point."""
    try:
        query = urllib.parse.urlencode({
            "lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10,
            "addressdetails": 1,
        })
        data = fetch_json("https://nominatim.openstreetmap.org/reverse?" + query)
        address = data.get("address", {})
        return (
            address.get("city") or address.get("town") or address.get("village")
            or address.get("municipality") or address.get("county") or fallback
        )
    except Exception:
        return fallback


def visibility_risk_points(visibility: float) -> tuple[int, str]:
    if visibility < 0.2:
        return 100, "Visibility is extremely low (below 200 m)."
    if visibility < 0.5:
        return 82, "Visibility is very low (below 500 m)."
    if visibility < 1.0:
        return 62, "Visibility is low (below 1 km)."
    if visibility < 2.0:
        return 35, "Visibility is slightly reduced."
    return 5, "Visibility is clear for normal road travel."


def rain_risk_points(rain: float) -> tuple[int, str]:
    if rain > 30:
        return 100, "Very heavy rainfall can make the road unsafe."
    if rain > 15:
        return 78, "Heavy rainfall may reduce control and visibility."
    if rain > 5:
        return 52, "Moderate rain may make the road slippery."
    if rain > 2:
        return 25, "Light rain is expected."
    return 3, "No significant rainfall is expected."


def wind_risk_points(wind: float) -> tuple[int, str]:
    if wind > 87:
        return 100, "Very strong wind may make travel unsafe."
    if wind > 62:
        return 75, "Strong wind may affect vehicle control."
    if wind > 40:
        return 48, "Wind is stronger than normal."
    if wind > 30:
        return 22, "A mild wind caution applies."
    return 3, "Wind speed is normal."


def condition_risk_points(features: dict, visibility: float) -> tuple[int, str]:
    humidity = float(features["humidity"])
    cloud = float(features["cloud_cover"])
    # Humidity/cloud are supporting factors only. They matter when visibility is already poor.
    if visibility < 1 and humidity >= 90:
        return 80, "High humidity with low visibility indicates possible fog."
    if visibility < 2 and humidity >= 85:
        return 48, "Humid conditions may cause haze or fog patches."
    if cloud >= 95 and float(features["precipitation"]) > 5:
        return 35, "Dense cloud with rain may reduce road clarity."
    return 4, "No major fog or storm indicator is present."


def time_risk_points(hour: int) -> tuple[int, str]:
    if hour >= 22 or hour < 5:
        return 55, "Night-time travel needs extra attention."
    if hour < 7 or hour >= 19:
        return 25, "Low-light conditions may slightly reduce road clarity."
    return 3, "Departure is during normal daylight hours."


def explain_checkpoint(features: dict, visibility: float, risk_level: str) -> tuple[str, list[str]]:
    checks = [
        visibility_risk_points(visibility),
        rain_risk_points(float(features["precipitation"])),
        wind_risk_points(float(features["wind_speed"])),
        condition_risk_points(features, visibility),
        time_risk_points(int(features["hour"])),
    ]
    important = sorted(checks, key=lambda x: x[0], reverse=True)
    if risk_level == "Low":
        reasons = [text for score, text in important if score <= 10][:3]
        reason = " ".join(reasons[:2]) or "Weather conditions are suitable for travel."
    else:
        reasons = [text for score, text in important if score >= 20][:3]
        reason = " ".join(reasons[:2]) or "Some route conditions require attention."
    return reason, reasons


def safety_from_rules(features: dict, visibility: float) -> dict:
    vis, _ = visibility_risk_points(visibility)
    rain, _ = rain_risk_points(float(features["precipitation"]))
    wind, _ = wind_risk_points(float(features["wind_speed"]))
    condition, _ = condition_risk_points(features, visibility)
    time_score, _ = time_risk_points(int(features["hour"]))
    risk_score = round(vis * .45 + rain * .25 + wind * .15 + condition * .10 + time_score * .05)
    safety = int(clamp(100 - risk_score, 0, 100))
    if risk_score < 18:
        risk = "Low"
    elif risk_score < 38:
        risk = "Mild"
    elif risk_score < 58:
        risk = "Medium"
    elif risk_score < 78:
        risk = "High"
    else:
        risk = "Very High"
    reason, reason_points = explain_checkpoint(features, visibility, risk)
    return {"risk_level": risk, "risk_score": risk_score, "safety_score": safety,
            "reason": reason, "reason_points": reason_points}


def route_recommendation(checkpoints: list[dict], worst: dict) -> dict:
    avg_safety = round(float(np.mean([p["safety_score"] for p in checkpoints])))
    max_risk = max(p["risk_score"] for p in checkpoints)
    low_count = sum(p["risk_level"] in {"High", "Very High"} for p in checkpoints)
    if max_risk >= 78 or avg_safety < 35:
        decision = "Avoid Travel"
        suggestion = "Postpone the journey until weather conditions improve."
    elif max_risk >= 58 or avg_safety < 60 or low_count:
        decision = "Travel with Caution"
        suggestion = f"Drive slowly near {worst['place_name']} and maintain a safe distance."
    else:
        decision = "Safe to Travel"
        suggestion = "Route conditions are generally suitable. Continue to follow normal road safety rules."

    positive = []
    warnings = []
    avg_rain = float(np.mean([p["features"]["precipitation"] for p in checkpoints]))
    avg_wind = float(np.mean([p["features"]["wind_speed"] for p in checkpoints]))
    avg_aqi = round(float(np.mean([p["features"]["aqi"] for p in checkpoints])))
    avg_visibility = float(np.mean([p["predicted_visibility_km"] for p in checkpoints]))
    if avg_visibility >= 2: positive.append("Visibility is clear across most of the route.")
    else: warnings.append("Visibility is reduced on parts of the route.")
    if avg_rain <= 2: positive.append("No heavy rainfall is expected.")
    else: warnings.append("Rain may affect road conditions.")
    if avg_wind <= 30: positive.append("Wind speed is normal.")
    else: warnings.append("Strong wind may affect vehicle control.")
    if avg_aqi > 200: warnings.append("Air quality is poor; this is a health advisory, not the main road-risk factor.")
    elif avg_aqi > 100: warnings.append("Air quality is moderate; sensitive travellers should take care.")
    else: positive.append("Air quality is acceptable.")

    return {
        "decision": decision,
        "reason": worst["reason"],
        "suggestion": suggestion,
        "positive_points": positive,
        "warning_points": warnings,
        "confidence_percent": round(float(np.mean([p["confidence_percent"] for p in checkpoints]))),
        "factors": {"average_rain_mm": round(avg_rain, 2),
                    "average_humidity_percent": round(float(np.mean([p["features"]["humidity"] for p in checkpoints])), 1),
                    "average_aqi": avg_aqi},
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/about")
def about():
    return render_template("about.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", system="ML-powered travel visibility assistant", time=datetime.now().isoformat())


@app.get("/api/model-info")
def model_info():
    return jsonify(json.loads(METRICS_PATH.read_text()))


@app.post("/api/predict")
def predict():
    try:
        return jsonify(predict_one(request.get_json(force=True)))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/analyze-route")
def analyze_route():
    try:
        body = request.get_json(force=True)
        source_text, destination_text = body["source"].strip(), body["destination"].strip()
        if not source_text or not destination_text:
            raise ValueError("Source and destination are required.")
        if source_text.lower() == destination_text.lower():
            raise ValueError("Source and destination must be different.")

        departure_text = body.get("departure")
        departure = datetime.fromisoformat(departure_text) if departure_text else datetime.now()
        source, destination = geocode(source_text), geocode(destination_text)
        distance = haversine(source["lat"], source["lon"], destination["lat"], destination["lon"])

        checkpoints = []
        checkpoint_count = 7
        used_names = set()
        for i in range(checkpoint_count):
            fraction = i / (checkpoint_count - 1)
            lat = source["lat"] + (destination["lat"] - source["lat"]) * fraction
            lon = source["lon"] + (destination["lon"] - source["lon"]) * fraction
            features = weather(lat, lon, i, departure)
            ml_prediction = predict_one(features)
            if i == 0:
                place = source["name"].split(",")[0]
            elif i == checkpoint_count - 1:
                place = destination["name"].split(",")[0]
            else:
                place = reverse_geocode(lat, lon, f"Route point {i + 1}")
            # Keep duplicate names understandable by adding distance only when needed.
            if place in used_names:
                place = f"Near {place} ({round(distance * fraction)} km)"
            used_names.add(place)
            rules = safety_from_rules(features, ml_prediction["predicted_visibility_km"])
            checkpoints.append({
                "index": i + 1, "place_name": place, "lat": lat, "lon": lon,
                "distance_km": round(distance * fraction, 1), "features": features,
                "predicted_visibility_km": ml_prediction["predicted_visibility_km"],
                "confidence_percent": ml_prediction["confidence_percent"],
                **rules,
            })

        avg_visibility = round(float(np.mean([p["predicted_visibility_km"] for p in checkpoints])), 2)
        avg_safety = round(float(np.mean([p["safety_score"] for p in checkpoints])))
        worst = max(checkpoints, key=lambda p: p["risk_score"])
        risk_order = {"Low": 0, "Mild": 1, "Medium": 2, "High": 3, "Very High": 4}
        route_risk = max((p["risk_level"] for p in checkpoints), key=lambda r: risk_order[r])
        rec = route_recommendation(checkpoints, worst)
        vehicle = body.get("vehicle", "Car")
        speed = {"Car": 55, "Bike": 45, "Bus": 42, "Walking": 5}.get(vehicle, 50)
        duration_minutes = round(distance / speed * 60)

        return jsonify({
            "source": source, "destination": destination, "vehicle": vehicle,
            "departure": departure.isoformat(timespec="minutes"),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "distance_km": round(distance, 1), "duration_minutes": duration_minutes,
            "average_visibility_km": avg_visibility, "risk_level": route_risk,
            "safety_score": avg_safety, "confidence_percent": rec["confidence_percent"],
            "worst_checkpoint": worst, "checkpoints": checkpoints,
            "recommendation": rec,
            "model": f"{BUNDLE['visibility_model_name']} visibility prediction model",
            "route_note": "Place names are obtained from route coordinates. The line is an approximate route and not turn-by-turn navigation.",
            "disclaimer": "This prediction is generated using a machine learning model. Weather can change unexpectedly, so use it as travel guidance rather than a guarantee.",
        })
    except Exception as exc:
        return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
