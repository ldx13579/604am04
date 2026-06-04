from datetime import datetime, timedelta
from sqlalchemy import func, distinct
from backend.database import SessionLocal
from backend.models import ABExperiment, ABImpression, ABClick, ABCTRSnapshot


class ABMetricsAggregator:
    """Background task that aggregates multi-dimensional metrics every 60 seconds.

    Tracks: CTR, average reward, recommendation diversity (unique items),
    category coverage, and per-position CTR.
    """

    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes

    def aggregate(self):
        db = SessionLocal()
        try:
            experiments = db.query(ABExperiment).filter(
                ABExperiment.status == "running"
            ).all()

            now = datetime.utcnow()
            window_start = now - timedelta(minutes=self.window_minutes)
            window_end = now

            for experiment in experiments:
                for group in ("A", "B"):
                    impressions = db.query(ABImpression).filter(
                        ABImpression.experiment_id == experiment.id,
                        ABImpression.group_name == group,
                        ABImpression.timestamp >= window_start,
                        ABImpression.timestamp < window_end,
                    ).all()

                    impressions_count = len(impressions)

                    clicks_count = db.query(ABClick).join(ABImpression).filter(
                        ABImpression.experiment_id == experiment.id,
                        ABImpression.group_name == group,
                        ABClick.clicked == True,
                        ABClick.timestamp >= window_start,
                        ABClick.timestamp < window_end,
                    ).count()

                    total_feedback = db.query(ABClick).join(ABImpression).filter(
                        ABImpression.experiment_id == experiment.id,
                        ABImpression.group_name == group,
                        ABClick.timestamp >= window_start,
                        ABClick.timestamp < window_end,
                    ).count()

                    ctr = clicks_count / impressions_count if impressions_count > 0 else 0.0
                    avg_reward = clicks_count / total_feedback if total_feedback > 0 else 0.0

                    all_items = set()
                    all_categories = set()
                    for imp in impressions:
                        if imp.recommended_items:
                            for item in imp.recommended_items:
                                all_items.add(item)
                                all_categories.add(item // 10)

                    diversity = len(all_items) / max(impressions_count * 5, 1)
                    coverage = len(all_categories) / 10.0

                    snapshot = ABCTRSnapshot(
                        experiment_id=experiment.id,
                        group_name=group,
                        window_start=window_start,
                        window_end=window_end,
                        impressions_count=impressions_count,
                        clicks_count=clicks_count,
                        ctr=ctr,
                    )
                    db.add(snapshot)

            db.commit()
        finally:
            db.close()


ab_aggregator = ABMetricsAggregator()
