from datetime import datetime, date, time
from pathlib import Path
import urllib.request
import importlib

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="RouteVision | Travel Planner",
    page_icon="🌍",
    layout="wide"
)

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "ml_system.joblib"

MODEL_URL = (
    "https://github.com/satwik2006/"
    "visibility-prediction-model/releases/download/"
    "v1.0/ml_system.joblib"
)


@st.cache_resource
def load_backend():
    MODEL_DIR.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        with st.spinner("Downloading trained ML model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    return importlib.import_module("app")


st.title("🌍 RouteVision")
st.subheader("AI-Powered Travel Visibility & Safety Planner")

st.write(
    "Plan your journey with machine-learning visibility "
    "predictions, weather analysis and route safety guidance."
)

try:
    backend = load_backend()
    st.success("Trained ML model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


st.divider()
st.subheader("Plan Your Journey")

col1, col2 = st.columns(2)

with col1:
    source = st.text_input(
        "Starting Location",
        placeholder="e.g. Prayagraj"
    )

with col2:
    destination = st.text_input(
        "Destination",
        placeholder="e.g. Varanasi"
    )

col3, col4, col5 = st.columns(3)

with col3:
    vehicle = st.selectbox(
        "Vehicle",
        ["Car", "Bike", "Bus", "Walking"]
    )

with col4:
    travel_date = st.date_input(
        "Departure Date",
        value=date.today()
    )

with col5:
    travel_time = st.time_input(
        "Departure Time",
        value=time(9, 0)
    )


if st.button("Analyze My Route", type="primary",
             use_container_width=True):

    if not source.strip() or not destination.strip():
        st.warning("Please enter both locations.")

    elif source.strip().lower() == destination.strip().lower():
        st.warning("Starting location and destination must be different.")

    else:
        try:
            departure = datetime.combine(
                travel_date, travel_time
            )

            with st.spinner("Analyzing your journey..."):

                source_data = backend.geocode(source)
                destination_data = backend.geocode(destination)

                distance = backend.haversine(
                    source_data["lat"],
                    source_data["lon"],
                    destination_data["lat"],
                    destination_data["lon"]
                )

                checkpoints = []
                used_names = set()

                for i in range(7):

                    fraction = i / 6

                    lat = (
                        source_data["lat"]
                        + (destination_data["lat"]
                           - source_data["lat"]) * fraction
                    )

                    lon = (
                        source_data["lon"]
                        + (destination_data["lon"]
                           - source_data["lon"]) * fraction
                    )

                    features = backend.weather(
                        lat, lon, i, departure
                    )

                    prediction = backend.predict_one(features)

                    if i == 0:
                        place = source_data["name"].split(",")[0]
                    elif i == 6:
                        place = destination_data["name"].split(",")[0]
                    else:
                        place = backend.reverse_geocode(
                            lat, lon, f"Route Point {i + 1}"
                        )

                    if place in used_names:
                        place = (
                            f"Near {place} "
                            f"({round(distance * fraction)} km)"
                        )

                    used_names.add(place)

                    rules = backend.safety_from_rules(
                        features,
                        prediction["predicted_visibility_km"]
                    )

                    checkpoints.append({
                        "index": i + 1,
                        "place_name": place,
                        "lat": lat,
                        "lon": lon,
                        "distance_km": round(distance * fraction, 1),
                        "features": features,
                        "predicted_visibility_km":
                            prediction["predicted_visibility_km"],
                        "confidence_percent":
                            prediction["confidence_percent"],
                        **rules
                    })

                worst = max(
                    checkpoints,
                    key=lambda p: p["risk_score"]
                )

                recommendation = backend.route_recommendation(
                    checkpoints, worst
                )

                avg_visibility = round(
                    sum(
                        p["predicted_visibility_km"]
                        for p in checkpoints
                    ) / 7, 2
                )

                avg_safety = round(
                    sum(p["safety_score"] for p in checkpoints) / 7
                )

                avg_confidence = recommendation["confidence_percent"]

                speed = {
                    "Car": 55,
                    "Bike": 45,
                    "Bus": 42,
                    "Walking": 5
                }[vehicle]

                duration = round(distance / speed * 60)

            st.divider()
            st.header("📊 Journey Analysis")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Distance", f"{distance:.1f} km")
            c2.metric("Estimated Time", f"{duration} min")
            c3.metric("Average Visibility", f"{avg_visibility} km")
            c4.metric("Safety Score", f"{avg_safety}/100")

            st.subheader("Travel Recommendation")

            decision = recommendation["decision"]

            if decision == "Safe to Travel":
                st.success(f"✅ {decision}")
            elif decision == "Travel with Caution":
                st.warning(f"⚠️ {decision}")
            else:
                st.error(f"⛔ {decision}")

            st.write(recommendation["suggestion"])

            st.write(
                f"**Worst checkpoint:** {worst['place_name']}"
            )

            st.write(
                f"**Prediction confidence:** {avg_confidence}%"
            )

            st.divider()
            st.subheader("🗺️ Route Checkpoints")

            map_data = pd.DataFrame([
                {
                    "latitude": p["lat"],
                    "longitude": p["lon"]
                }
                for p in checkpoints
            ])

            st.map(map_data)

            table_data = []

            for p in checkpoints:
                table_data.append({
                    "Checkpoint": p["index"],
                    "Place": p["place_name"],
                    "Distance (km)": p["distance_km"],
                    "Visibility (km)": p["predicted_visibility_km"],
                    "Risk": p["risk_level"],
                    "Safety Score": p["safety_score"],
                    "Confidence (%)": p["confidence_percent"],
                    "Reason": p["reason"]
                })

            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True
            )

            st.divider()
            st.subheader("✅ Positive Factors")

            for item in recommendation["positive_points"]:
                st.write("•", item)

            st.subheader("⚠️ Warnings")

            if recommendation["warning_points"]:
                for item in recommendation["warning_points"]:
                    st.write("•", item)
            else:
                st.write("No major warnings detected.")

            st.caption(
                "Weather data may use an offline demonstration "
                "fallback if live weather is unavailable."
            )

        except Exception as e:
            st.error(f"Route analysis failed: {e}")


st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:15px;
        color:gray;
        font-size:15px;
    ">
        <b>Created by Satwik Shukla</b>
    </div>
    """,
    unsafe_allow_html=True
)
