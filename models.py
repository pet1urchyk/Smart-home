from datetime import datetime
from sqlalchemy import String, Float, ForeignKey, DateTime, func, Index,Integer,Boolean,LargeBinary
from sqlalchemy.dialects.postgresql import JSONB,ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

class Sensor(Base):
    __tablename__ = "sensors"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    type: Mapped[str] =mapped_column(String(50), server_default="unknown", nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    location: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    firmware_version: Mapped[str] = mapped_column(String(25),server_default="v.1.0.0")

    events: Mapped[list["Sensor_event"]] = relationship(
        "Sensor_event",
        back_populates="sensor",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    anomalies: Mapped[list["AnomalyScore"]] = relationship(
        "AnomalyScore",
        back_populates="sensor",
        cascade="all,delete-orphan",
        passive_deletes=True
    )

class Sensor_event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "ix_sensor_event_payload_gin",
            "payload",
            postgresql_using="gin"
        ),
    )

    sensor_type: Mapped[str] = mapped_column(String(50),nullable=False)
    sensor_id: Mapped[int] = mapped_column(ForeignKey("sensors.id", ondelete="CASCADE"), nullable=False)
    sensor: Mapped["Sensor"] = relationship("Sensor", back_populates="events")
    anomaly: Mapped["AnomalyScore | None"] = relationship(
       "AnomalyScore", uselist=False, back_populates="event", cascade="all, delete-orphan",passive_deletes=True)

class BaselineStats(Base):
    __tablename__ = "baseline_stats"

    id: Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    sensor_id: Mapped[int] = mapped_column(ForeignKey("sensors.id", ondelete="CASCADE"),nullable=False)

    mean: Mapped[float] = mapped_column(Float, nullable=False)
    std: Mapped[float] = mapped_column(Float,nullable=True)

    model: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    window_minutes: Mapped[int] = mapped_column(Integer,nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer,nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
class AnomalyScore(Base):
    __tablename__ = "anomaly_scores"

    id: Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"),nullable=False)
    sensor_id: Mapped[int] = mapped_column(ForeignKey("sensors.id",ondelete="CASCADE"),nullable=False)

    delta: Mapped[float] = mapped_column(Float,nullable=False)
    z_score: Mapped[float] = mapped_column(Float,nullable=True)
    if_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    sus: Mapped[bool] = mapped_column(Boolean,nullable=False)

    sensor: Mapped["Sensor"] = relationship("Sensor",back_populates="anomalies")
    event: Mapped["Sensor_event"] = relationship(
       "Sensor_event", back_populates="anomaly"
   )
    
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Prediction_model(Base):
    __tablename__ = "prediction_model"

    id: Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    sensor_id: Mapped[int] = mapped_column(ForeignKey("sensors.id", ondelete="CASCADE"),nullable=False)

    mae: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)

    model_type: Mapped[str] = mapped_column(String,nullable=False)
    eval_mode: Mapped[str] = mapped_column(String,nullable=False)
    model: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    last_source_id: Mapped[int] = mapped_column(Integer,server_default="67")

    window_minutes: Mapped[int] = mapped_column(Integer,nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer,nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class PredictScore(Base):
    __tablename__ = "predictscore"

    id: Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    sensor_id: Mapped[int] = mapped_column(ForeignKey("sensors.id",ondelete="CASCADE"),nullable=False) 
    prediction_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("prediction_model.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("events.id", ondelete="SET NULL"),
        nullable=True,)

    model_type: Mapped[str] = mapped_column(String,nullable=False)
    eval_mode: Mapped[str] = mapped_column(String,nullable=False)

    steps: Mapped[int] = mapped_column(Integer,nullable=False)

    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
class PredictRun(Base):
    __tablename__ = "predictrun"

    id: Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("predictscore.id",ondelete="CASCADE"),nullable=False)

    horizon_step: Mapped[int] = mapped_column(Integer, nullable=False)
    y_hat: Mapped[float] = mapped_column(Float, nullable=False)