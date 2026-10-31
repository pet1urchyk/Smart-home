import models
from datetime import datetime,timezone,timedelta

from sqlalchemy import select

from database import AsyncSessionLocal
import asyncio

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

import os
import pickle
async def main(limit=1000):
    while True:
        minutes = 36000
        async with AsyncSessionLocal() as db:
            timelimit = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            stmt = (
                select(models.Sensor_event)
                .where(models.Sensor_event.sensor_type == "temp_hum")
                .where(models.Sensor_event.sensor_id == 2)
                .where(models.Sensor_event.timestamp >= timelimit)
                .order_by(models.Sensor_event.timestamp.asc())
                .limit(limit)
            )
            
            result = await db.execute(stmt)
            events = result.scalars().all()
            if not events:
                continue
            sensor_id = events[0].sensor_id
            temps = [float(e.payload["temperature"]) for e in events]
            hums = [float(e.payload["humidity"]) for e in events]
            print(len(hums))
            print(hums[:5])
            print(hums[-5:])
            print(len(temps))
            print(temps[:5]) 
            print(temps[-5:]) 
            lags = []
            true_temps = []
            for i in range(2, len(temps)):
                lag1 = temps[i - 1]
                lag2 = temps[i - 2]
                delta = temps[i-1] - temps[i-2]
                lags.append((lag1,lag2,delta))
                true_temps.append(temps[i])
            X = np.array(lags)
            y = np.array(true_temps)
            print(X.shape, y.shape)
            print(X[:3])
            print(y[:3])
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            if len(X_test) < 180:
                raise ValueError(f"Need at least 180 test samples, got {len(X_test)}")
            lags_h = []
            true_hums = []
            for i in range(2,len(hums)):
                lag1 = hums[i-1]
                lag2 = hums[i-2]
                delta = hums[i-1] - hums[i-2]
                lags_h.append((lag1,lag2,delta))
                true_hums.append(hums[i])
            X_h = np.array(lags_h)
            y_h = np.array(true_hums)
            print(X.shape,y.shape)
            print(X[:3])
            print(y[:3])
            split_idx = int(len(X_h) * 0.8)
            X_h_train,X_h_test = X_h[:split_idx],X_h[split_idx:]
            y_h_train,y_h_test = y_h[:split_idx],y_h[split_idx:]

            if len(X_h_test) < 180:
                raise ValueError(f"Need at least 180 test samples, got {len(X_h_test)}")
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            lag1 = X_test[0, 0]
            lag2 = X_test[0, 1]
            delta = X_test[0,2]

            predictions = []

            for _ in range(180):
                X_current = np.array([[lag1, lag2,delta]])

                pred = model.predict(X_current)[0]
                predictions.append(pred)

                lag2 = lag1
                lag1 = pred
                delta = lag1 - lag2

            predictions = np.array(predictions)
            print(predictions)

            MAE_rd_180 = np.mean(np.abs(y_test[:180] - predictions))
            MAE_rd_60 = np.mean(np.abs(y_test[:60] - predictions[:60]))
            MAE_rd_12 = np.mean(np.abs(y_test[:12] - predictions[:12]))
            MAE_rd_1 = np.mean(np.abs(y_test[:1] - predictions[:1]))
            
            print(f"Ridge MAE 180 steps: {MAE_rd_180}")
            print(f"Ridge MAE 60 steps: {MAE_rd_60}")
            print(f"Ridge MAE 12 steps: {MAE_rd_12}")
            print(f"Ridge MAE 1 step: {MAE_rd_1}")

            model_h = Ridge(alpha=1.0)
            model_h.fit(X_h_train,y_h_train)

            lag1_h = X_h_test[0, 0]
            lag2_h = X_h_test[0, 1]
            delta_h = X_h_test[0,2]

            predictions = []

            for _ in range(180):
                X_current_h = np.array([[lag1_h, lag2_h,delta_h]])

                pred = model_h.predict(X_current_h)[0]
                predictions.append(pred)

                lag2_h = lag1_h
                lag1_h = pred
                delta_h = lag1_h - lag2_h

            predictions = np.array(predictions)
            print(predictions)

            MAE_rd_180_h = np.mean(np.abs(y_h_test[:180] - predictions))
            MAE_rd_60_h = np.mean(np.abs(y_h_test[:60] - predictions[:60]))
            MAE_rd_12_h = np.mean(np.abs(y_h_test[:12] - predictions[:12]))
            MAE_rd_1_h = np.mean(np.abs(y_h_test[:1] - predictions[:1]))
            
            print(f"Ridge MAE 180 steps humidity: {MAE_rd_180_h}")
            print(f"Ridge MAE 60 steps humidity: {MAE_rd_60_h}")
            print(f"Ridge MAE 12 steps humidity: {MAE_rd_12_h}")
            print(f"Ridge MAE 1 step humidity: {MAE_rd_1_h}")

            trees = 100
            seed = 42
            cores = -1
            forest = RandomForestRegressor(
                n_estimators=trees,
                random_state=seed,
                n_jobs=cores
            )
            forest.fit(X_train, y_train)

            lag1 = X_test[0, 0]
            lag2 = X_test[0, 1]
            delta = X_test[0,2]

            forest_predictions = []

            for _ in range(180):
                X_current = np.array([[lag1, lag2,delta]])

                pred = forest.predict(X_current)[0]
                forest_predictions.append(pred)

                lag2 = lag1
                lag1 = pred
                delta = lag1 - lag2

            forest_predictions = np.array(forest_predictions)

            MAE_rf_180 = np.mean(np.abs(y_test[:180] - forest_predictions))
            MAE_rf_60 = np.mean(np.abs(y_test[:60] - forest_predictions[:60]))
            MAE_rf_12 = np.mean(np.abs(y_test[:12] - forest_predictions[:12]))
            MAE_rf_1 = np.mean(np.abs(y_test[:1] - forest_predictions[:1]))
            
            print(f"Random Forest MAE 180 steps: {MAE_rf_180}")
            print(f"Random Forest MAE 60 steps: {MAE_rf_60}")
            print(f"Random Forest MAE 12 steps: {MAE_rf_12}")
            print(f"Random Forest MAE 1 step: {MAE_rf_1}")

            forest_h = RandomForestRegressor(
                n_estimators=trees,
                random_state=seed,
                n_jobs = cores
            )
            forest_h.fit(X_h_train,y_h_train)

            forest_h_predictions = []

            lag1_h = X_h_test[0,0]
            lag2_h = X_h_test[0,1]
            delta_h = X_h_test[0,2]

            for _ in range(180):
                X_current_h = np.array([[lag1_h,lag2_h,delta_h]])

                pred = forest_h.predict(X_current_h)[0]
                forest_h_predictions.append(pred)

                lag2_h = lag1_h
                lag1_h = pred
                delta_h = lag1_h-lag2_h

            forest_predictions = np.array(forest_h_predictions)

            MAE_rf_180_h = np.mean(np.abs(y_h_test[:180] - forest_predictions))
            MAE_rf_60_h = np.mean(np.abs(y_h_test[:60] - forest_predictions[:60]))
            MAE_rf_12_h = np.mean(np.abs(y_h_test[:12] - forest_predictions[:12]))
            MAE_rf_1_h = np.mean(np.abs(y_h_test[:1] - forest_predictions[:1]))
            
            print(f"Random Forest MAE humidity 180 steps: {MAE_rf_180_h}")
            print(f"Random Forest MAE humidity 60 steps: {MAE_rf_60_h}")
            print(f"Random Forest MAE humidity 12 steps: {MAE_rf_12_h}")
            print(f"Random Forest MAE humidity 1 step: {MAE_rf_1_h}")

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            mlflow.set_experiment(f"sensor-{sensor_id}-predict")

            client = MlflowClient()

            scores = {
                "random_forest_180":MAE_rf_180,
                "random_forest_60":MAE_rf_60,
                "random_forest_12":MAE_rf_12,
                "random_forest_1":MAE_rf_1,

                "ridge_180":MAE_rd_180,
                "ridge_60":MAE_rd_60,
                "ridge_12":MAE_rd_12,
                "ridge_1":MAE_rd_1,
            }
            forest_MAE = 0
            forest_MAE_docs = []

            ridge_MAE = 0
            ridge_MAE_docs = []

            for key, mae in scores.items():
                if "random_forest" in key:
                    forest_MAE += float(mae)
                    forest_MAE_docs.append(mae)
                if "ridge" in key:
                    ridge_MAE += float(mae)
                    ridge_MAE_docs.append(mae)

            winner = "forest" if forest_MAE < ridge_MAE else "ridge"
            maes_winner_docs = forest_MAE_docs if winner == "forest" else ridge_MAE_docs
            compare_winner = forest_MAE if forest_MAE < ridge_MAE else ridge_MAE

            print("Scores:",scores)
            print("Winner:",winner)
            if winner == "ridge":
                ridge_prod = model
                ridge_prod.fit(X,y)
                model_type = "ridge"
                eval_mode = "recursive_180"
                with mlflow.start_run(run_name=f"ridge-model-predict-{sensor_id}") as run:
                    mlflow.log_param("sensor_id",sensor_id)
                    mlflow.log_param("window_minutes",minutes)
                    mlflow.log_param("sample_count",len(X)) 
                    mlflow.log_param("model_type",model_type)
                    mlflow.log_param("eval_mode",eval_mode)
                    
                    info = mlflow.sklearn.log_model(
                        sk_model=ridge_prod,
                        name="model"
                    )

                    run_id = run.info.run_id
                    model_winner = ridge_prod
            else:
                forest_prod = forest
                forest_prod.fit(X,y)
                model_type = "random_forest"
                eval_mode = "recursive_180"
                with mlflow.start_run(run_name=f"random_forest-model-predict-{sensor_id}") as run:
                    mlflow.log_param("sensor_id",sensor_id)
                    mlflow.log_param("window_minutes",minutes)
                    mlflow.log_param("cores-active",cores)
                    mlflow.log_param("trees",trees)
                    mlflow.log_param("seed",seed)
                    mlflow.log_param("sample_count",len(X)) 
                    mlflow.log_param("model_type",model_type)
                    mlflow.log_param("eval_mode",eval_mode)
                                    
                    mlflow.log_metric("MAE", MAE_rf_180)

                    info = mlflow.sklearn.log_model(
                        sk_model=forest_prod,
                        name="model"
                    )
            
                    run_id = run.info.run_id
                    model_winner = forest_prod

            name = "sensor_2_predict"
            try:
                client.create_registered_model(name)
            except Exception:
                pass

            mv = client.create_model_version(
                name=name,
                source=info.model_uri,
                run_id=run_id
            )            
            candidate_version = mv.version

            try:
                champion_info = client.get_model_version_by_alias(
                    name,
                    "champion"
                )
                champion_version = champion_info.version

                run = client.get_run(champion_info.run_id)
                params = run.data.params

                champion_type = params.get("model_type") or "unknown"
                champion_eval = params.get("eval_mode") or "recursive_180"

                champion_model = mlflow.sklearn.load_model(
                    champion_info.source
                    )

                lag1 = X_test[0,0]
                lag2 = X_test[0,1]
                delta = X_test[0,2]

                predictions = []
                for _ in range(180):
                    X_current = np.array([[lag1,lag2,delta]])

                    pred = champion_model.predict(X_current)[0]
                    predictions.append(pred)

                    lag2 = lag1
                    lag1 = pred
                    delta = lag1 - lag2 

                predictions = np.array(predictions)
                print(predictions)

                MAE_champion_180 = np.mean(np.abs(y_test[:180] - predictions))
                MAE_champion_60 = np.mean(np.abs(y_test[:60] - predictions[:60]))
                MAE_champion_12 = np.mean(np.abs(y_test[:12] - predictions[:12]))
                MAE_champion_1 = np.mean(np.abs(y_test[:1] - predictions[:1]))
                
                print(f"Champion MAE 180 steps: {MAE_champion_180}")
                print(f"Champion MAE 60 steps: {MAE_champion_60}")
                print(f"Champion MAE 12 steps: {MAE_champion_12}")
                print(f"Champion MAE 1 step: {MAE_champion_1}")

            except Exception as e:
                print(f"No champion available: {e}")
                MAE_champion_180 = float("inf")
                MAE_champion_60 = float("inf")
                MAE_champion_12 = float("inf")
                MAE_champion_1 = float("inf")
                champion_version = None
                champion_model = None

            scores_champion = {
                "champion_180":MAE_champion_180,
                "champion_60":MAE_champion_60,
                "champion_12":MAE_champion_12,
                "champion_1":MAE_champion_1,
            }

            MAE_champion = 0
            champion_MAE_docs = []
            for key, mae in scores_champion.items():
                    MAE_champion += float(mae)
                    champion_MAE_docs.append(mae)

            if compare_winner >= MAE_champion:
                model_winner = champion_model
                eval_mode = champion_eval
                model_type = champion_type
            else:
                client.set_registered_model_alias(
                name=name,
                alias="champion",
                version=str(candidate_version)
            )
                champion_version = candidate_version
                champion_MAE_docs = maes_winner_docs
            
            print("Champion version:",champion_version)
            print("Candidate version:",candidate_version)
            print()
            print("Champion MAE:",MAE_champion)
            print("Candidate MAE:",maes_winner_docs)
            print()     

            if model_winner is None:
                continue

            model_predict = models.Prediction_model(
                sensor_id=sensor_id,

                mae=champion_MAE_docs,
                model=pickle.dumps(model_winner),
                model_type=model_type,
                eval_mode=eval_mode,

                window_minutes=minutes,
                sample_count=len(X)
            )
            db.add(model_predict)

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            mlflow.set_experiment(f"sensor-{sensor_id}-predict-hum")

            client = MlflowClient()

            scores = {
                "random_forest_180":MAE_rf_180_h,
                "random_forest_60":MAE_rf_60_h,
                "random_forest_12":MAE_rf_12_h,
                "random_forest_1":MAE_rf_1_h,

                "ridge_180":MAE_rd_180_h,
                "ridge_60":MAE_rd_60_h,
                "ridge_12":MAE_rd_12_h,
                "ridge_1":MAE_rd_1_h,
            }
            forest_MAE = 0
            forest_MAE_docs = []

            ridge_MAE = 0
            ridge_MAE_docs = []

            for key, mae in scores.items():
                if "random_forest" in key:
                    forest_MAE += float(mae)
                    forest_MAE_docs.append(mae)
                if "ridge" in key:
                    ridge_MAE += float(mae)
                    ridge_MAE_docs.append(mae)

            winner = "forest" if forest_MAE < ridge_MAE else "ridge"
            maes_winner_docs = forest_MAE_docs if winner == "forest" else ridge_MAE_docs
            compare_winner = forest_MAE if forest_MAE < ridge_MAE else ridge_MAE

            print("Scores:",scores)
            print("Winner:",winner)
            if winner == "ridge":
                ridge_prod_h = model_h
                ridge_prod_h.fit(X_h,y_h)
                model_type = "ridge_h"
                eval_mode = "recursive_180"
                with mlflow.start_run(run_name=f"ridge-model-predict-{sensor_id}-h") as run:
                    mlflow.log_param("sensor_id",sensor_id)
                    mlflow.log_param("window_minutes",minutes)
                    mlflow.log_param("sample_count",len(X_h)) 
                    mlflow.log_param("model_type",model_type)
                    mlflow.log_param("eval_mode",eval_mode)
                    
                    mlflow.log_metric("MAE",MAE_rd_180_h)
                    
                    info = mlflow.sklearn.log_model(
                        sk_model=ridge_prod_h,
                        name="model"
                    )

                    run_id = run.info.run_id
                    model_winner = ridge_prod_h
            else:
                forest_prod_h = forest_h
                forest_prod_h.fit(X_h,y_h)
                model_type = "random_forest_h"
                eval_mode = "recursive_180"
                with mlflow.start_run(run_name=f"random_forest-model-predict-{sensor_id}-h") as run:
                    mlflow.log_param("sensor_id",sensor_id)
                    mlflow.log_param("window_minutes",minutes)
                    mlflow.log_param("cores-active",cores)
                    mlflow.log_param("trees",trees)
                    mlflow.log_param("seed",seed)
                    mlflow.log_param("sample_count",len(X_h)) 
                    mlflow.log_param("model_type",model_type)
                    mlflow.log_param("eval_mode",eval_mode)
                                    
                    mlflow.log_metric("MAE", MAE_rf_180_h)

                    info = mlflow.sklearn.log_model(
                        sk_model=forest_prod_h,
                        name="model"
                    )
            
                    run_id = run.info.run_id
                    model_winner = forest_prod_h

            name = "sensor_2_predict_h"
            try:
                client.create_registered_model(name)
            except Exception:
                pass

            mv = client.create_model_version(
                name=name,
                source=info.model_uri,
                run_id=run_id
            )            
            candidate_version = mv.version

            try:
                champion_info = client.get_model_version_by_alias(
                    name,
                    "champion"
                )
                champion_version_h = champion_info.version

                run = client.get_run(champion_info.run_id)
                params = run.data.params

                champion_type = params.get("model_type") or "unknown"
                champion_eval = params.get("eval_mode") or "recursive_180"

                champion_model_h = mlflow.sklearn.load_model(
                    champion_info.source
                    )

                lag1_h = X_h_test[0, 0]
                lag2_h = X_h_test[0, 1]
                delta_h = X_h_test[0, 2]

                predictions = []
                for _ in range(180):
                    X_current = np.array([[lag1_h,lag2_h,delta_h]])

                    pred = champion_model_h.predict(X_current)[0]
                    predictions.append(pred)

                    lag2_h = lag1_h
                    lag1_h = pred
                    delta_h = lag1_h - lag2_h 

                predictions = np.array(predictions)
                print(predictions)

                MAE_champion_180 = np.mean(np.abs(y_h_test[:180] - predictions))
                MAE_champion_60 = np.mean(np.abs(y_h_test[:60] - predictions[:60]))
                MAE_champion_12 = np.mean(np.abs(y_h_test[:12] - predictions[:12]))
                MAE_champion_1 = np.mean(np.abs(y_h_test[:1] - predictions[:1]))
                
                print(f"Champion MAE humidity 180 steps: {MAE_champion_180}")
                print(f"Champion MAE humidity 60 steps: {MAE_champion_60}")
                print(f"Champion MAE humidity 12 steps: {MAE_champion_12}")
                print(f"Champion MAE humidity 1 step: {MAE_champion_1}")

            except Exception as e:
                print(f"No champion available: {e}")
                MAE_champion_180 = float("inf")
                MAE_champion_60 = float("inf")
                MAE_champion_12 = float("inf")
                MAE_champion_1 = float("inf")
                champion_version_h = None
                champion_model_h = None

            scores_champion = {
                "champion_180":MAE_champion_180,
                "champion_60":MAE_champion_60,
                "champion_12":MAE_champion_12,
                "champion_1":MAE_champion_1,
            }

            MAE_champion = 0
            champion_MAE_docs = []
            for key, mae in scores_champion.items():
                    MAE_champion += float(mae)
                    champion_MAE_docs.append(mae)

            if compare_winner >= MAE_champion:
                model_winner = champion_model_h
                eval_mode = champion_eval
                model_type = champion_type
                champion_version_h = champion_version_h
            else:
                client.set_registered_model_alias(
                name=name,
                alias="champion",
                version=str(candidate_version)
            )
                champion_version_h = candidate_version
                champion_MAE_docs = maes_winner_docs
            
            print("Champion version:",champion_version_h)
            print("Candidate version:",candidate_version)
            print()
            print("Champion MAE:",MAE_champion)
            print("Candidate MAE:",maes_winner_docs)
            print()     

            if model_winner is None:
                continue    

            model_predict_h = models.Prediction_model(
                sensor_id=sensor_id,

                mae=champion_MAE_docs,
                model=pickle.dumps(model_winner),
                model_type=model_type,
                eval_mode=eval_mode,

                window_minutes=minutes,
                sample_count=len(X_h)
            )
            
            db.add(model_predict_h)
            await db.commit()
            print("YES!")
            await asyncio.sleep(15*60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Connection closed")