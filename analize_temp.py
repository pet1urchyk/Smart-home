import models
from datetime import datetime,timezone,timedelta

from sqlalchemy import select

from database import AsyncSessionLocal
import asyncio

import pickle
import os

import numpy as np
from sklearn.ensemble import IsolationForest   

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

TEST_CASES = [
    {
        "name": "high_temperature",
        "input": (30.0, 50.0),
        "expected": -1,
    },
    {
        "name": "low_temperature",
        "input": (14.0, 50.0),
        "expected": -1,
        },
    {
        "name": "high_humidity",
        "input": (22.0, 80.0),
        "expected": -1,
    },
    {
        "name": "low_humidity",
        "input": (24.0, 20.0),
        "expected": -1,
        },
    {
        "name": "normal_conditions",
        "input": (22.0, 50.0),
        "expected": 1,
    },
    {
        "name": "normal_conditions3",
        "input": (24.0, 55.0),
        "expected": 1,
    },
    {
        "name": "normal_conditions2",
        "input": (20.0, 45.0),
        "expected": 1,
    },
]
async def get_event(limit=1000):
     while True:
        async with AsyncSessionLocal() as db:
            time_limit = datetime.now(timezone.utc) - timedelta(minutes=180)

            stmt = select(models.Sensor_event
            ).where(models.Sensor_event.sensor_type == "temp_hum"
            ).where(models.Sensor_event.timestamp >= time_limit
            ).order_by(models.Sensor_event.timestamp.desc()
            ).limit(limit)

            result = await db.execute(stmt)
            events = result.scalars().all()

            events_by_sensor = {}

            for event in events:
                try:
                    sensor_id = int(event.sensor_id)
                    temp = float(event.payload["temperature"])
                    hum = float(event.payload["humidity"])
                    
                    if sensor_id not in events_by_sensor:
                        events_by_sensor[sensor_id] = []
                    
                    events_by_sensor[sensor_id].append((temp, hum))
                except (KeyError, ValueError, TypeError, AttributeError):
                    continue

            for sensor_id,payload in events_by_sensor.items():

                if len(payload) <= 100:
                    print(f"Sensor {sensor_id}: Not enough data ({len(payload)})")
                    continue

                arr = np.array(payload,dtype=float)
                print(arr.shape)
                
                print("we have a ",len(arr),"amount of temperatures")
                print("the first 5:",arr[:5])
                print("the last 5:",arr[-5:])

                temp_col = arr[:, 0]
                hum_col = arr[:, 1]

                med_t = float(np.median(temp_col))
                mad_t = float(np.median(np.abs(temp_col - med_t)))
                sigma_t = (mad_t / 0.6745) if mad_t > 0 else 0.0

                med_h = float(np.median(hum_col))
                mad_h = float(np.median(np.abs(hum_col - med_h)))
                sigma_h = (mad_h / 0.6745) if mad_h > 0 else 0.0

                good_rows = []
                for temp, hum in arr:
                    z_t = (temp - med_t) / sigma_t if sigma_t > 1e-12 else None
                    z_h = (hum - med_h) / sigma_h if sigma_h > 1e-12 else None
                    
                    is_outlier = (z_t is not None and abs(z_t) > 3) or (z_h is not None and abs(z_h) > 3)
                    
                    if not is_outlier:
                        good_rows.append((temp, hum))        
                if len(good_rows) <= 100:
                    print(f"Sensor {sensor_id}: Not enough clean data ({len(good_rows)})")
                    continue
                good_rows = np.array(good_rows, dtype=float)    
                X = good_rows.reshape(-1,2)
                print(X.shape)
                print("YES!")
                mean = float(np.mean(good_rows[:,0]))
                std = float(np.std(good_rows[:,0]))
                
                model = IsolationForest(contamination=0.0033,random_state=42,n_estimators=200)
                model.fit(X)

                name=f"sk_learn-IF-anomaly-{sensor_id}"
                version_alias = "champion"

                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
                mlflow.set_experiment("sensor-isolation-forest")

                with mlflow.start_run(run_name=f"if-sensor{sensor_id}-c{0.0033}") as run:
                    mlflow.log_param("contamination", 0.0033)
                    mlflow.log_param("sensor_id", sensor_id)
                    mlflow.log_param("window_minutes", 180)
                    mlflow.log_metric("sample_count", len(good_rows))
                    mlflow.log_metric("mean", mean)
                    mlflow.log_metric("std", std)
                    run_id = run.info.run_id
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name=f"sklearn-model-IF-{sensor_id}",
                        input_example=np.array([[30.0,50.0]]),
                        registered_model_name=f"sk_learn-IF-anomaly-{sensor_id}"
                        )
                    
                client = MlflowClient()
                def evaluate_model(model):
                    results = []
                    passed = 0

                    for test in TEST_CASES:

                        temp,hum = test["input"]

                        X = np.array([(temp,hum)],dtype=float)
                        print(X.shape)
                        prediction = model.predict(X)[0]
                        success = prediction == test["expected"]

                        if success:
                            passed += 1

                        results.append({
                            "name": test["name"],
                            "prediction": int(prediction),
                            "expected": test["expected"],
                            "passed": success
                        })
                    score = passed / len(TEST_CASES)

                    return score,results
                candidate_score, candidate_results = evaluate_model(model)


                versions = client.search_model_versions(
                    f"name='sk_learn-IF-anomaly-{sensor_id}'"
                )
                new_version = next((
                    version
                    for version in versions
                    if version.run_id == run_id
                ),None)
                if new_version is None:
                    raise RuntimeError(
                        f"Model version for run {run_id} war not found"
                    )
                candidate_version = new_version.version


                try:
                    champion_info = client.get_model_version_by_alias(
                        name,
                        "champion"
                    )
                    champion_version = champion_info.version

                    champion_uri = f"models:/{name}@champion"
                    champion_model = mlflow.sklearn.load_model(
                        champion_uri
                    )
                    champion_score, champion_results = evaluate_model(champion_model)

                except Exception:
                    champion_score = -1
                    champion_version = None
                    champion_results = []

                if champion_score >= candidate_score:
                    pass
                else:
                    client.set_registered_model_alias(
                        name,
                        "champion",
                        candidate_version
                    )
                    champion_version = candidate_version
                    champion_score = candidate_score
                    champion_results = candidate_results
                    
                print("Champion version:",champion_version)
                print("Candidate version:",candidate_version)
                print()
                print("Champion results:",champion_results)
                print("Candidate results:",candidate_results)
                print()
                print("Champion score:",champion_score)
                print("Candidate score:", candidate_score)                

                model_info = client.get_model_version_by_alias(name, version_alias)
                model_tags = model_info.tags
                print(model_tags)

                model_uri = f"models:/{name}@{version_alias}"
                model = mlflow.sklearn.load_model(model_uri)
                serialized_model = pickle.dumps(model)

                print(f"std:{std}")
                print(f"mean:{mean}")

                new_anomaly = models.BaselineStats(
                    sensor_id=sensor_id,
                    mean=mean,
                    std=std,
                    window_minutes=180,
                    sample_count=len(good_rows),
                    model=serialized_model
                )
                db.add(new_anomaly)

            await db.commit()    
        await asyncio.sleep(180*60)


if __name__ == "__main__":
    try:
        asyncio.run(get_event())
    except KeyboardInterrupt:
        print("Connection closed")