import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import app, predict_one

PAYLOAD = {
    "temperature": 28, "humidity": 70, "pressure": 1012,
    "wind_speed": 8, "precipitation": 0, "cloud_cover": 40,
    "aqi": 75, "hour": 12, "month": 7,
}


class Tests(unittest.TestCase):
    def test_multi_model_prediction(self):
        result = predict_one(PAYLOAD)
        self.assertGreater(result["predicted_visibility_km"], 0)
        self.assertIn(result["risk_level"], ["Low", "Moderate", "High", "Severe"])
        self.assertIn(result["departure_action"], ["Leave Now", "Travel with Caution", "Delay if Possible"])
        self.assertGreaterEqual(result["safety_score"], 0)
        self.assertLessEqual(result["safety_score"], 100)
        self.assertEqual(len(result["models"]), 4)

    def test_predict_api(self):
        client = app.test_client()
        response = client.post("/api/predict", json=PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertIn("departure_action", response.get_json())

    @patch("app.weather")
    def test_route_api(self, mock_weather):
        mock_weather.return_value = PAYLOAD | {"data_source": "test"}
        client = app.test_client()
        response = client.post("/api/analyze-route", json={
            "source": "Prayagraj", "destination": "Jhusi", "vehicle": "Car",
            "departure": "2026-07-21T12:00",
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data["checkpoints"]), 7)
        self.assertIn("80%", data["ml_dependency"])
        self.assertLessEqual(data["worst_checkpoint"]["distance_km"], data["distance_km"])


if __name__ == "__main__":
    unittest.main()
