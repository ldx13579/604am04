import uuid
import numpy as np
from datetime import datetime
from fastapi import APIRouter, HTTPException
from backend.database import SessionLocal
from backend.models import ABExperiment, ABImpression, ABClick, OnlineInteraction
from backend.services.policy_loader import policy_loader, traffic_allocator
from backend.schemas import (
    RecommendRequest, RecommendResponse, FeedbackRequest, PolicyInfoResponse,
    CacheConfigUpdateRequest,
)

router = APIRouter(prefix="/api/recommend", tags=["recommend"])


@router.post("/predict", response_model=RecommendResponse)
async def predict(request: RecommendRequest):
    if len(request.user_state) != 10:
        raise HTTPException(status_code=400, detail="user_state must be 10-dimensional")

    state = np.array(request.user_state, dtype=np.float32)
    session_id = request.session_id or str(uuid.uuid4())
    group = None
    experiment_id = None

    db = SessionLocal()
    try:
        experiment = db.query(ABExperiment).filter(
            ABExperiment.status == "running"
        ).first()

        if experiment:
            group = traffic_allocator.assign_group(
                session_id, experiment.id, experiment.traffic_split
            )
            experiment_id = experiment.id

            if group == "A":
                top_k = policy_loader.get_top_k(state, request.top_k)
            else:
                items = np.random.choice(100, size=request.top_k, replace=False)
                top_k = [(int(i), 0.0) for i in items]
        else:
            top_k = policy_loader.get_top_k(state, request.top_k)

        items = [item_id for item_id, _ in top_k]
        scores = [score for _, score in top_k]

        impression = ABImpression(
            experiment_id=experiment_id or 0,
            group_name=group or "A",
            user_state=request.user_state,
            recommended_items=items,
            session_id=session_id,
            timestamp=datetime.utcnow(),
        )
        if experiment_id:
            db.add(impression)
            db.commit()
            db.refresh(impression)
            impression_id = impression.id
        else:
            impression_id = None

        return RecommendResponse(
            items=items,
            scores=scores,
            policy_version_id=policy_loader.current_version_id,
            group=group,
            impression_id=impression_id,
        )
    finally:
        db.close()


@router.post("/feedback")
async def feedback(request: FeedbackRequest):
    db = SessionLocal()
    try:
        click = ABClick(
            impression_id=request.impression_id,
            item_id=request.item_id,
            clicked=request.clicked,
            timestamp=datetime.utcnow(),
        )
        db.add(click)

        if request.impression_id:
            impression = db.query(ABImpression).get(request.impression_id)
            if impression:
                state = impression.user_state
                next_state = state[:]
                category = request.item_id // 10
                if request.clicked and category < len(next_state):
                    next_state[category] = min(1.0, next_state[category] + 0.1)

                interaction = OnlineInteraction(
                    state=state,
                    action=request.item_id,
                    reward=1.0 if request.clicked else 0.0,
                    next_state=next_state,
                    done=False,
                    timestamp=datetime.utcnow(),
                )
                db.add(interaction)

        db.commit()
        return {"status": "recorded"}
    finally:
        db.close()


@router.get("/policy_info", response_model=PolicyInfoResponse)
async def policy_info():
    return PolicyInfoResponse(
        is_loaded=policy_loader.is_loaded,
        policy_version_id=policy_loader.current_version_id,
        algorithm=policy_loader.current_algorithm,
    )


@router.get("/traffic_balance")
async def traffic_balance():
    db = SessionLocal()
    try:
        experiment = db.query(ABExperiment).filter(
            ABExperiment.status == "running"
        ).first()
        if not experiment:
            return {"active": False}
        balance = traffic_allocator.get_balance(experiment.id)
        balance["active"] = True
        balance["target_split_a"] = experiment.traffic_split
        return balance
    finally:
        db.close()


@router.get("/cache_config")
async def get_cache_config():
    return policy_loader.cache.stats


@router.put("/cache_config")
async def update_cache_config(request: CacheConfigUpdateRequest):
    policy_loader.cache.update_config(
        mode=request.mode,
        ttl_seconds=request.ttl_seconds,
    )
    if request.invalidate:
        policy_loader.invalidate_cache()
    return policy_loader.cache.stats
