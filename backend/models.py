from sqlalchemy import Column, Integer, BigInteger, Float, Boolean, String, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship
from backend.database import Base


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, nullable=False, index=True)
    popularity = Column(Float, nullable=False, default=1.0)
    embedding = Column(ARRAY(Float), nullable=False)


class OfflineTransition(Base):
    __tablename__ = "offline_transitions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    state = Column(ARRAY(Float), nullable=False)
    action = Column(Integer, nullable=False)
    reward = Column(Float, nullable=False)
    next_state = Column(ARRAY(Float), nullable=False)
    done = Column(Boolean, nullable=False)
    episode_id = Column(Integer, nullable=False, index=True)
    timestamp_step = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm = Column(String(50), nullable=False, index=True)
    hyperparameters = Column(JSONB, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)
    total_epochs = Column(Integer, nullable=False, default=0)
    best_reward = Column(Float, nullable=True)

    metrics = relationship("TrainingMetric", back_populates="run", cascade="all, delete-orphan")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    q_value_mean = Column(Float, nullable=True)
    q_value_max = Column(Float, nullable=True)
    q_value_min = Column(Float, nullable=True)
    cumulative_reward = Column(Float, nullable=False)
    cql_penalty = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    run = relationship("TrainingRun", back_populates="metrics")
