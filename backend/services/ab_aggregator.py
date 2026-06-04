from datetime import datetime, timedelta
from sqlalchemy import func
from backend.database import SessionLocal
from backend.models import ABExperiment, ABImpression, ABClick, ABCTRSnapshot


class ABMetricsAggregator:
    """Background task that aggregates CTR metrics every 60 seconds."""

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
                    impressions_count = db.query(ABImpression).filter(
                        ABImpression.experiment_id == experiment.id,
                        ABImpression.group_name == group,
                        ABImpression.timestamp >= window_start,
                        ABImpression.timestamp < window_end,
                    ).count()

                    clicks_count = db.query(ABClick).join(ABImpression).filter(
                        ABImpression.experiment_id == experiment.id,
                        ABImpression.group_name == group,
                        ABClick.clicked == True,
                        ABClick.timestamp >= window_start,
                        ABClick.timestamp < window_end,
                    ).count()

                    ctr = clicks_count / impressions_count if impressions_count > 0 else 0.0

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
