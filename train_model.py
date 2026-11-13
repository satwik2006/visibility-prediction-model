from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dataset' / 'GlobalWeatherRepository.csv'
MODEL_DIR = ROOT / 'model'
FEATURES = ['temperature','humidity','pressure','wind_speed','precipitation','cloud_cover','aqi','hour','month']
AUX_FEATURES = FEATURES + ['predicted_visibility_km']
SOURCE_COLUMNS = ['temperature_celsius','humidity','pressure_mb','wind_kph','precip_mm','cloud','air_quality_us-epa-index','last_updated','visibility_km']

def reg_metrics(y,p):
    return {'MAE':round(float(mean_absolute_error(y,p)),4),'RMSE':round(float(mean_squared_error(y,p)**0.5),4),'R2':round(float(r2_score(y,p)),4)}
def cls_metrics(y,p):
    return {'accuracy':round(float(accuracy_score(y,p)),4),'weighted_f1':round(float(f1_score(y,p,average='weighted')),4)}

def load_real_data():
    if not DATA.exists(): raise FileNotFoundError(f'Missing real dataset: {DATA}')
    raw = pd.read_csv(DATA, usecols=SOURCE_COLUMNS)
    dt = pd.to_datetime(raw['last_updated'], errors='coerce')
    df = pd.DataFrame({
        'temperature':raw['temperature_celsius'], 'humidity':raw['humidity'], 'pressure':raw['pressure_mb'],
        'wind_speed':raw['wind_kph'], 'precipitation':raw['precip_mm'], 'cloud_cover':raw['cloud'],
        'aqi':raw['air_quality_us-epa-index'], 'hour':dt.dt.hour, 'month':dt.dt.month,
        'visibility_km':raw['visibility_km']
    }).replace([np.inf,-np.inf],np.nan).dropna()
    # ML targets for the travel assistant are generated from observed visibility plus recorded environment.
    severe = (df.visibility_km < 2) | (df.precipitation >= 10)
    high = (df.visibility_km < 5) | (df.precipitation >= 5) | (df.aqi >= 5)
    moderate = (df.visibility_km < 8) | (df.humidity >= 90) | (df.aqi >= 4)
    df['risk_label'] = np.select([severe,high,moderate],['Severe','High','Moderate'],default='Low')
    penalty = (np.maximum(0,10-df.visibility_km)*5 + df.precipitation*2.5 + np.maximum(0,df.humidity-80)*0.35 + np.maximum(0,df.aqi-2)*4)
    df['safety_score'] = np.clip(100-penalty,0,100)
    df['departure_action'] = np.select([df.safety_score < 35, df.safety_score < 60, df.safety_score < 78],['Avoid Travel','Delay Journey','Travel with Caution'],default='Safe to Travel')
    # Keep training practical while preserving uncommon visibility values.
    if len(df) > 40000:
        common = df[df.visibility_km == 10].sample(22000, random_state=42)
        uncommon = df[df.visibility_km != 10]
        df = pd.concat([common, uncommon], ignore_index=True).sample(frac=1, random_state=42)
    return df

def main():
    df=load_real_data()
    train_df,test_df=train_test_split(df,test_size=.2,random_state=42,stratify=df['risk_label'])
    Xtr,Xte=train_df[FEATURES],test_df[FEATURES]; ytr,yte=train_df.visibility_km,test_df.visibility_km
    candidates={
      'Linear Regression':LinearRegression(),
      'Decision Tree':DecisionTreeRegressor(max_depth=16,min_samples_leaf=4,random_state=42),
      'Random Forest':RandomForestRegressor(n_estimators=70,max_depth=16,min_samples_leaf=2,random_state=42,n_jobs=-1),
      'Gradient Boosting':GradientBoostingRegressor(n_estimators=80,learning_rate=.06,max_depth=3,random_state=42)}
    results={}; fitted={}
    for name,est in candidates.items():
        steps=[('imputer',SimpleImputer(strategy='median'))]
        if name=='Linear Regression': steps.append(('scaler',StandardScaler()))
        steps.append(('model',est)); pipe=Pipeline(steps).fit(Xtr,ytr)
        results[name]=reg_metrics(yte,pipe.predict(Xte)); fitted[name]=pipe
    best=max(results,key=lambda n:results[n]['R2']); vis=fitted[best]
    tr_aug=Xtr.copy(); te_aug=Xte.copy(); tr_aug['predicted_visibility_km']=vis.predict(Xtr); te_aug['predicted_visibility_km']=vis.predict(Xte)
    risk=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=70,max_depth=14,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1))]).fit(tr_aug,train_df.risk_label)
    safety=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',GradientBoostingRegressor(n_estimators=70,learning_rate=.06,max_depth=3,random_state=42))]).fit(tr_aug,train_df.safety_score)
    departure=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',GradientBoostingClassifier(n_estimators=60,learning_rate=.06,max_depth=3,random_state=42))]).fit(tr_aug,train_df.departure_action)
    errors=np.abs(yte.to_numpy()-vis.predict(Xte)); q50,q90=np.quantile(errors,[.5,.9])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({'visibility_model':vis,'risk_model':risk,'safety_model':safety,'departure_model':departure,'features':FEATURES,'aux_features':AUX_FEATURES,'visibility_model_name':best,'confidence_error_q50':float(q50),'confidence_error_q90':float(q90)},MODEL_DIR/'ml_system.joblib')
    meta={'dataset':'Global Weather Repository (real-world daily weather observations)','dataset_file':'GlobalWeatherRepository.csv','source':'Kaggle - Global Weather Repository by Nelgiriyewithana','rows_original':154556,'rows_used':len(df),'ml_dependency_estimate':'Approximately 80%+ of core prediction and decision logic','deployed_models':{'visibility':best,'risk':'Random Forest Classifier','safety_score':'Gradient Boosting Regressor','departure_recommendation':'Gradient Boosting Classifier','confidence':'Classifier probability + held-out regression error calibration'},'visibility_model_comparison':results,'auxiliary_metrics':{'risk_classifier':cls_metrics(test_df.risk_label,risk.predict(te_aug)),'safety_regressor':reg_metrics(test_df.safety_score,safety.predict(te_aug)),'departure_classifier':cls_metrics(test_df.departure_action,departure.predict(te_aug))},'features':FEATURES,'target':'visibility_km','note':'Visibility is an observed dataset column. Risk, safety and departure labels are engineered from observed visibility and recorded environmental conditions for educational decision-support modelling.'}
    (MODEL_DIR/'model_metrics.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
