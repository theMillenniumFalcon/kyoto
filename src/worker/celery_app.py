"""
Celery application — broker and result backend both use Redis.

Tasks are defined here so they can be imported by both the worker
process and the API (for `.delay()` dispatch).
"""
from celery import Celery
from src.config.settings import settings

celery_app = Celery(
    "kyoto",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["src.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Route all ingestion jobs to a dedicated queue
    task_routes={"src.worker.tasks.ingest_repo": {"queue": "ingest"}},
    # Prevent tasks from silently swallowing exceptions
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker process
)