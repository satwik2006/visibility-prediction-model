<img width="1366" height="768" alt="dashboard" src="https://github.com/user-attachments/assets/14b05002-dd75-4150-b750-2bee23dfcd84" />
# Visibility Prediction Model — Real Dataset Edition

A Flask-based travel visibility assistant powered by machine-learning models trained using the **Global Weather Repository** dataset.

## Dataset
- File: `dataset/GlobalWeatherRepository.csv`
- Source: Kaggle, **Global Weather Repository** by Nelgiriyewithana
- Original records: 154,556
- Observed target: `visibility_km`
- Inputs: temperature, humidity, pressure, wind speed, precipitation, cloud cover, AQI, hour and month

Risk, safety score and departure recommendation labels are engineered from observed visibility and environmental measurements for educational decision-support modelling.

## Run
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```
Open `http://127.0.0.1:5000`.

## ML system
The training script compares Linear Regression, Decision Tree, Random Forest and Gradient Boosting for visibility, deploys the best test-set model, and trains auxiliary ML models for risk, safety score and travel recommendation.

This is an educational project, not a substitute for official weather or road-safety advisories.

## RouteVision travel-first update

- Homepage redesigned as a journey planner instead of an ML dashboard.
- Checkpoints now display locality/place names using reverse geocoding.
- Added an easy-language **Reason** column for every place.
- Added route-level positive factors, warnings and a clear recommendation.
- Added an ML prediction disclaimer below the result.
- Visibility remains predicted by the trained ML model.
- Travel risk uses transparent weighted thresholds: visibility 45%, rain 25%, wind 15%, fog/storm indicators 10%, departure time 5%.
- Technical model comparison is available only on the About Project page.


## 🚀 Live Demo

[Click Here to Open RouteVision]
(https://visibility-prediction-model-423jx9ov3fs2fzzwhqr9yq.streamlit.app/)
