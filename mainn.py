from contextlib import asynccontextmanager
from typing import Optional, List, Literal, Union,Annotated
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Query, Depends, status,Response
from pydantic import BaseModel, Field, ConfigDict, model_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from database import engine, Base, get_db
import models

import json
import redis_pool
import pickle

from sklearn.ensemble import IsolationForest
import numpy as np

import asyncio

@asynccontextmanager
async def lifespan(app:FastAPI):
    await redis_pool.redis_client.ping()
    yield

app = FastAPI(title="Sensor API",lifespan=lifespan)

class SensorCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)  
    location: str = Field(default="Сенсор в приміщенні", max_length=200)

class SensorPut(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)  
    location: str = Field(default="Сенсор в приміщенні", max_length=200)
    status: str    

class SensorUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None

class Sensor(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    location: str
    status: str = "active"
    created_at: datetime

class Temp_Hum(BaseModel):
    sensor_type: Literal["temp_hum"]
    temperature: float
    humidity: float = Field(..., ge=0, le=100)

class Motion(BaseModel):
    sensor_type: Literal["motion"]
    detected: bool 

SensorEventCreate = Annotated[
    Union[Motion, Temp_Hum],
    Field(discriminator="sensor_type")
]

class SensorEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sensor_id: int
    sensor_type: str
    payload: SensorEventCreate

    timestamp: datetime

    delta: float | None = None
    z_score: float | None = None
    if_score: float | None = None
    sus: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def extract_anomaly(cls,data):
        if hasattr(data,"anomaly") and data.anomaly:
            data_dict = {
                "id": data.id,
                "sensor_id": data.sensor_id,
                "sensor_type": data.sensor_type,
                "payload": data.payload,
                "timestamp": data.timestamp,
                "delta": data.anomaly.delta,
                "z_score": data.anomaly.z_score,
                "if_score": data.anomaly.if_score,
                "sus": data.anomaly.sus,
            }
            return data_dict
        return data

class SensorEventPredict(BaseModel):
    run_id:int
    sensor_id:int
    horizont_y_hat: dict[int,float]

async def invalidate_events_cache(sensor_id: int) -> None:
    await redis_pool.redis_client.delete(f"sensors:{sensor_id}:events")

    pattern = f"sensors:{sensor_id}:events:*"
    keys = [key async for key in redis_pool.redis_client.scan_iter(match=pattern)]
    if keys:
        await redis_pool.redis_client.delete(*keys)

HUMIDITY_TYPES = ("ridge_humidity","random_forest_humidity")
TEMPERATURE_TYPES = ("ridge_temperature","random_forest_temperature")

@app.get("/sensors", response_model=List[Sensor])
async def location_sensors(
    location: str | None = None, 
    db: AsyncSession = Depends(get_db)
):
    cached_key = f"sensors:location:{location}" if location else "sensors:all"
    cached_data = await redis_pool.redis_client.get(cached_key)
    if cached_data:
        return Response(content=cached_data, media_type="application/json")
    
    stmt = select(models.Sensor)
    if location is not None:
        stmt = stmt.where(models.Sensor.location == location) 

    result = await db.execute(stmt)
    sensors = result.scalars().all()

    sensors_dict = [Sensor.model_validate(s).model_dump(mode="json") for s in sensors]
    json_str = json.dumps(sensors_dict)
    await redis_pool.redis_client.set(cached_key, json_str, ex=60)
    
    return Response(content=json_str, media_type="application/json")

@app.post("/sensors",response_model=Sensor,status_code=201)
async def create_sensor(data:SensorCreate,db:AsyncSession = Depends(get_db)) -> Sensor:
    
    new_sensor = models.Sensor(
        name=data.name,
        location=data.location,
    )
    db.add(new_sensor)
    
    await db.commit()
    await db.refresh(new_sensor)

    await redis_pool.redis_client.delete("sensors:all")
    await redis_pool.redis_client.delete(f"sensors:location:{new_sensor.location}")

    return new_sensor 

@app.get("/events/recent", response_model=List[SensorEvent])
async def event_now(
    minutes: int = Query(..., gt=0, le=1440),
    db:AsyncSession = Depends(get_db),
    limit: int = Query(default=100, le=1000)
    ):
    time_limit = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    
    stmt = (
        select(models.Sensor_event)
        .options(joinedload(models.Sensor_event.anomaly)) 
        .where(models.Sensor_event.timestamp >= time_limit)
        .order_by(models.Sensor_event.timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    actual_event = result.scalars().all()

    return actual_event
    
@app.get("/events", response_model=List[SensorEvent], status_code=200)
async def events_list(
    limit: int = Query(default=100, le=1000), 
    db: AsyncSession = Depends(get_db)
):
    stmt = (
        select(models.Sensor_event)
        .options(joinedload(models.Sensor_event.anomaly)) 
        .order_by(models.Sensor_event.timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    events = result.scalars().all() 

    return events

@app.put("/sensors/{id}", response_model=Sensor)
async def update_sensor(id: int, data: SensorPut, db:AsyncSession = Depends(get_db)):
    stmt = select(models.Sensor).where(models.Sensor.id == id)
    result = await db.execute(stmt)
    sensor = result.scalar_one_or_none()

    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    old_location = sensor.location

    sensor.name = data.name
    sensor.location = data.location
    sensor.status = data.status

    await db.commit()
    await db.refresh(sensor)

    await redis_pool.redis_client.delete("sensors:all")
    await redis_pool.redis_client.delete(f"sensors:id:{id}")
    await redis_pool.redis_client.delete(f"sensors:location:{old_location}")
    await redis_pool.redis_client.delete(f"sensors:location:{sensor.location}")

    return sensor 

@app.get("/sensors/{id}",response_model=Sensor)
async def get_sensor(id: int,db:AsyncSession = Depends(get_db)) -> Sensor:
    cached_key = f"sensors:id:{id}"
    cached_data = await redis_pool.redis_client.get(cached_key)
    
    if cached_data:
        return Response(content=cached_data, media_type="application/json")

    stmt = select(models.Sensor).where(models.Sensor.id == id)
    result = await db.execute(stmt)
    sensor = result.scalar_one_or_none()
    if not sensor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sensor not found"
        )

    sensor_dict = Sensor.model_validate(sensor).model_dump(mode="json")
    json_str = json.dumps(sensor_dict)
    
    await redis_pool.redis_client.set(cached_key, json_str, ex=60)
    
    return Response(content=json_str, media_type="application/json")  

@app.patch("/sensors/{id}",response_model=Sensor)
async def patch_sensor(
    id: int,
    data:SensorUpdate,
    db:AsyncSession = Depends(get_db)):

    stmt = select(models.Sensor).where(models.Sensor.id == id)
    result = await db.execute(stmt)
    sensor = result.scalar_one_or_none()

    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    old_location = sensor.location

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(sensor, key, value) 

    await db.commit()
    await db.refresh(sensor)

    await redis_pool.redis_client.delete("sensors:all")
    await redis_pool.redis_client.delete(f"sensors:id:{id}")
    await redis_pool.redis_client.delete(f"sensors:location:{old_location}")
    await redis_pool.redis_client.delete(f"sensors:location:{sensor.location}")

    return sensor 

@app.delete("/sensors/{id}", status_code=204)
async def delete_sensor(id: int,db:AsyncSession = Depends(get_db)):
    stmt = select(models.Sensor).where(models.Sensor.id == id)
    result = await db.execute(stmt)
    sensor = result.scalar_one_or_none()

    if not sensor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Sensor not found")
    
    await db.delete(sensor)
    await db.commit()

    await redis_pool.redis_client.delete("sensors:all")
    await redis_pool.redis_client.delete(f"sensors:id:{id}")
    await redis_pool.redis_client.delete(f"sensors:location:{sensor.location}")
    await invalidate_events_cache(id)

    return None 

@app.delete("/events/{id}",status_code=204)
async def delete_event(id:int,db:AsyncSession = Depends(get_db)):
    stmt = select(models.Sensor_event).where(models.Sensor_event.id == id)
    result = await db.execute(stmt)
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )
    sensor_id = event.sensor_id

    await db.delete(event)
    await db.commit()

    await invalidate_events_cache(sensor_id)

    return None 

@app.post("/sensors/{sensor_id}/events", response_model=SensorEvent, status_code=201)
async def add_event(
    sensor_id: int, 
    data: SensorEventCreate,
    db:AsyncSession = Depends(get_db)):

    def predictss(event1,event2,delta,model,range_todo):
        predictions = []
        for _ in range(range_todo):
            X_current = np.array([[event1,event2,delta]])
            prediction = model.predict(X_current)[0]
            predictions.append(prediction)
            event2 = event1
            event1 = prediction
            delta = event1 - event2
                         
        return predictions
    
    stmt = select(models.Sensor).where(models.Sensor.id == sensor_id)
    result = await db.execute(stmt)
    sensor = result.scalar_one_or_none()

    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    new_event = models.Sensor_event(
        sensor_type = data.sensor_type,
        payload = data.model_dump(),
        sensor = sensor 
    )

    db.add(new_event)
    await db.flush()

    delta = None
    z = None
    if_score = None
    sus = None

    if data.sensor_type == "temp_hum":
        try:
            stmt = (
                    select(models.BaselineStats)
                    .where(models.BaselineStats.sensor_id == sensor_id)
                    .order_by(models.BaselineStats.computed_at.desc()) 
                    .limit(1)
            )
            result = await db.execute(stmt)
            baseline = result.scalars().first()

            if baseline and baseline.model:
                temp = data.temperature
                hum = data.humidity

                results = []

                raw_delta = temp - baseline.mean
                delta = abs(raw_delta)
                delta_result = True if delta > 4 else False
                results.append(delta_result)

                z = delta / baseline.std if baseline.std > 0 else 0
                z_score = True if z>2.5 else False
                results.append(z_score)

                def run_inference(model_bytes, temp,hum):
                    model: IsolationForest = pickle.loads(bytes(model_bytes))
                    features = np.array([[temp,hum]],dtype=float)
                    features = features.reshape(1,-1)
                    return float(model.decision_function(features)[0])

                if_score = await asyncio.to_thread(run_inference, baseline.model, temp,hum)
                    
                sus_if = False if if_score >= 0 else True 
                results.append(sus_if)

                sus = []
                for result in results:
                    if result == True:
                        sus.append(result)
                    else:
                        continue
                sus = True if len(sus) >= 2 else False            
                new_anomaly = models.AnomalyScore(
                    event_id=new_event.id,
                    sensor_id=new_event.sensor_id,
                    delta=delta,
                    z_score=z,
                    if_score=if_score, 
                    sus=sus
                )

                db.add(new_anomaly)
        except Exception as e:
            print("we had a problem:",e)    
        try:
            stmt = (
                    select(models.Prediction_model)
                    .where(models.Prediction_model.sensor_id == sensor_id)
                    .where(models.Prediction_model.model_type.in_(TEMPERATURE_TYPES))
                    .order_by(models.Prediction_model.computed_at.desc()) 
                    .limit(1)
                    )
            result = await db.execute(stmt)
            predict = result.scalars().first()
            stmt = (
                select(models.PredictScore.predicted_at)
                .where(models.PredictScore.sensor_id == sensor_id)
                .where(models.PredictScore.model_type.in_(TEMPERATURE_TYPES))
                .order_by(models.PredictScore.predicted_at.desc())
                .limit(1)
            )
            result = await db.execute(stmt)
            last = result.scalar_one_or_none()
            now = datetime.now(timezone.utc)
            if last == None or (now-last) >= timedelta(minutes=3):
                if predict and predict.model:
                    lag1 = data.temperature
                    stmt = (select(models.Sensor_event)
                            .where(models.Sensor_event.sensor_id == sensor_id)
                            .where(models.Sensor_event.sensor_type == "temp_hum")
                            .where(models.Sensor_event.id != new_event.id)
                            .order_by(models.Sensor_event.timestamp.desc())
                            .limit(1)
                            )
                    result = await db.execute(stmt)
                    lag2 = result.scalars().first()
                    lag2 = float(lag2.payload["temperature"])
                    delta = lag1-lag2
                    model = pickle.loads(predict.model)

                    predictscore = models.PredictScore(
                        sensor_id=new_event.sensor_id,
                        prediction_model_id=predict.id,
                        source_event_id=new_event.id,
                        model_type=predict.model_type,
                        eval_mode=predict.eval_mode,
                        steps=180
                    )
                    db.add(predictscore)
                    await db.flush()
                    
                    predicts = await asyncio.to_thread(predictss,lag1,lag2,delta,model,180)
                    for i, y in enumerate(predicts, start=1):
                        db.add(models.PredictRun(
                            run_id=predictscore.id,
                            horizon_step=i,
                            y_hat=y,
                        ))
                    pattern = f"sensors:{sensor_id}:events:forecast:*"
                    keys = [key async for key in redis_pool.redis_client.scan_iter(match=pattern)]
                    if keys:    
                        await redis_pool.redis_client.delete(*keys)
                                
        except Exception as e:
            print("we had a problem",e)
        try:
            stmt = (
                    select(models.Prediction_model)
                    .where(models.Prediction_model.sensor_id == sensor_id)
                    .where(models.Prediction_model.model_type.in_(HUMIDITY_TYPES))
                    .order_by(models.Prediction_model.computed_at.desc()) 
                    .limit(1)
                    )
            result = await db.execute(stmt)
            predict = result.scalars().first()

            stmt = (
                select(models.PredictScore.predicted_at)
                .where(models.PredictScore.sensor_id == sensor_id)
                .where(models.PredictScore.model_type.in_(HUMIDITY_TYPES))
                .order_by(models.PredictScore.predicted_at.desc())
                .limit(1)
            )
            result = await db.execute(stmt)
            last = result.scalar_one_or_none()
            
            now = datetime.now(timezone.utc)
            
            if last == None or (now-last) >= timedelta(minutes=3):
                if predict and predict.model:
                    lag1 = data.humidity
                    stmt = (select(models.Sensor_event)
                            .where(models.Sensor_event.sensor_id == sensor_id)
                            .where(models.Sensor_event.sensor_type == "temp_hum")
                            .where(models.Sensor_event.id != new_event.id)
                            .order_by(models.Sensor_event.timestamp.desc())
                            .limit(1)
                            )
                    result = await db.execute(stmt)
                    lag2 = result.scalars().first()
                    lag2 = float(lag2.payload["humidity"])
                    delta = lag1-lag2
                    model = pickle.loads(predict.model)

                    predictscore = models.PredictScore(
                        sensor_id=new_event.sensor_id,
                        prediction_model_id=predict.id,
                        source_event_id=new_event.id,
                        model_type=predict.model_type,
                        eval_mode=predict.eval_mode,
                        steps=180
                    )
                    db.add(predictscore)
                    await db.flush()    
                    predicts = await asyncio.to_thread(predictss,lag1,lag2,delta,model,180)
                    for i, y in enumerate(predicts, start=1):
                        db.add(models.PredictRun(
                            run_id=predictscore.id,
                            horizon_step=i,
                            y_hat=y,
                        ))
                    pattern = f"sensors:{sensor_id}:events:forecast:*"
                    keys = [key async for key in redis_pool.redis_client.scan_iter(match=pattern)]
                    if keys:    
                        await redis_pool.redis_client.delete(*keys)
                                
        except Exception as e:
            print("we had a problem",e)

    await db.commit()
    await db.refresh(new_event)
    await invalidate_events_cache(sensor_id)

    return SensorEvent(
        id=new_event.id,
        sensor_id=new_event.sensor_id,
        sensor_type=new_event.sensor_type,
        payload=data,  
        timestamp=new_event.timestamp,
        delta=delta,
        z_score=z,
        if_score=if_score,
        sus=sus
    )

@app.get("/sensors/{id}/events",response_model=List[SensorEvent],status_code=200)
async def event(
    id:int,
    db:AsyncSession=Depends(get_db),
    limit: int = Query(default=100,le=1000)
    ):
    
    cached_key = f"sensors:{id}:events:limit:{limit}" 
    cached_data = await redis_pool.redis_client.get(cached_key)

    if cached_data:
        return Response(content=cached_data, media_type="application/json")

    sensor_stmt = select(models.Sensor).where(models.Sensor.id == id)
    sensor_result = await db.execute(sensor_stmt)
    if not sensor_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Sensor not found")

    stmt = (
        select(models.Sensor_event)
        .options(joinedload(models.Sensor_event.anomaly)) 
        .where(models.Sensor_event.sensor_id == id)
        .order_by(models.Sensor_event.timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    readings = result.scalars().all()

    readings_dict = [SensorEvent.model_validate(r).model_dump(mode="json") for r in readings]
    json_str = json.dumps(readings_dict)
    
    await redis_pool.redis_client.set(cached_key, json_str, ex=30)
    return Response(content=json_str, media_type="application/json")

@app.get("/sensors/{id}/events/recent",response_model=List[SensorEvent])
async def event_now_sensor(
    id:int,minutes: int = Query(..., gt=0, le=1440),
    db:AsyncSession = Depends(get_db)
    ):
    cached_key = f"sensors:{id}:events:{minutes}"
    cached_data = await redis_pool.redis_client.get(cached_key)

    if cached_data:
        return Response(content=cached_data,media_type="application/json")
    
    stmt2 = select(models.Sensor).where(models.Sensor.id == id)
    result2 = await db.execute(stmt2)
    sensor = result2.scalar_one_or_none()
    
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    time_limit = datetime.now(timezone.utc) - timedelta(minutes=minutes)

    stmt = (select(models.Sensor_event)
        .options(joinedload(models.Sensor_event.anomaly)) 
        .where(models.Sensor_event.sensor_id == id,
        models.Sensor_event.timestamp >= time_limit).
        order_by(models.Sensor_event.timestamp.desc()))

    result = await db.execute(stmt)
    reading = result.scalars().all()

    reading_dict = [SensorEvent.model_validate(r).model_dump(mode="json") for r in reading]
    json_str = json.dumps(reading_dict)
    
    await redis_pool.redis_client.set(cached_key, json_str, ex=30)
    return Response(content=json_str, media_type="application/json")

@app.get("/sensors/{id}/events/forecast",response_model=SensorEventPredict)
async def forecast(
    id:int,
    db:AsyncSession = Depends(get_db),
    minutes: int = Query(15,gt=0,le=15),
    target: Literal["humidity","temperature"] = Query("temperature")
    ):
    
    stmt2 = select(models.Sensor).where(models.Sensor.id == id)
    result2 = await db.execute(stmt2)
    sensor = result2.scalar_one_or_none()
        
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    if target == "temperature":
        types = ("ridge_temperature", "random_forest_temperature")
    else:
        types = ("ridge_humidity", "random_forest_humidity")

    stmt = (select(models.PredictScore)
            .where(models.PredictScore.sensor_id == id)
            .where(models.PredictScore.model_type.in_(types))
            .order_by(models.PredictScore.predicted_at.desc())
            .limit(1))
    
    result = await db.execute(stmt)
    prediction = result.scalars().first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Forecast not found")
    
    run_id = prediction.id
    steps = minutes*12

    cached_key = f"sensors:{id}:events:forecast:{run_id}"
    cached_data = await redis_pool.redis_client.get(cached_key)

    if cached_data:
        if steps != 180:
            data = json.loads(cached_data)
            y_hat = data["horizont_y_hat"]
            horizont_y_hat = {
                int(k):float(v)
                for k,v in y_hat.items()
                if int(k) <= steps
            }      
            forecast_data = SensorEventPredict(
                run_id=run_id,
                sensor_id=id,
                horizont_y_hat=horizont_y_hat
            )
            json_str = forecast_data.model_dump_json()
            return Response(content=json_str,media_type="application/json")
        
        return Response(content=cached_data,media_type="application/json")

    stmt = (select(models.PredictRun)
            .where(models.PredictRun.run_id == run_id))
    results = await db.execute(stmt)
    predictions = results.scalars().all()
    
    horizon_map = {pred.horizon_step: pred.y_hat for pred in predictions}

    forecast_data = SensorEventPredict(
        run_id=run_id,
        sensor_id=id,
        horizont_y_hat=horizon_map
        )      
    json_str = forecast_data.model_dump_json()
    await redis_pool.redis_client.set(cached_key,json_str,ex=160)

    if steps != 180:
        horizon_map_r = {pred.horizon_step: pred.y_hat for pred in predictions if int(pred.horizon_step) <= steps}
        forecast_data = SensorEventPredict(
            run_id=run_id,
            sensor_id=id,
            horizont_y_hat=horizon_map_r
        ) 
        json_str_r = forecast_data.model_dump_json()
        return Response(content=json_str_r,media_type="application/json")

    return Response(content=json_str,media_type="application/json")