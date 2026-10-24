import logging
import uuid
from datetime import datetime, timedelta, timezone

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

from app.config import settings
from app.db import engine
from app.utils.time import now_utc

logger = logging.getLogger(__name__)

RSS_JOB_ID = "rss-poll-global"

_WEEKDAY_ABBR = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def config_job_id(config_id) -> str:
    return f"config-{config_id}"


def manual_job_id(config_id) -> str:
    return f"config-{config_id}-manual-{int(now_utc().timestamp())}"


scheduler = AsyncIOScheduler(
    jobstores={"default": SQLAlchemyJobStore(engine=engine)},
    job_defaults={"misfire_grace_time": 300, "coalesce": True, "max_instances": 1},
)


def start() -> None:
    if not scheduler.running:
        scheduler.start()
        logger.info("scheduler started")

    scheduler.add_job(
        "app.services.rss:poll_all_feeds",
        trigger=IntervalTrigger(hours=settings.rss_poll_interval_hours),
        id=RSS_JOB_ID,
        replace_existing=True,
    )

    # Rehydrate per-config jobs from the DB so the source of truth is always
    # newspaper_configs.active, not whatever the jobstore happens to hold.
    from sqlmodel import Session, select
    from app.models import NewspaperConfig

    with Session(engine) as s:
        active = s.exec(
            select(NewspaperConfig).where(NewspaperConfig.active == True)  # noqa: E712
        ).all()
        for cfg in active:
            try:
                register_config(cfg)
            except Exception:
                logger.exception("scheduler: failed to register config %s on startup", cfg.id)


def shutdown() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)


def _next_utc_anchor(run_hour: int) -> datetime:
    """Return the next UTC datetime at run_hour:00:00. If the current time
    is before run_hour today, returns today; otherwise tomorrow. Used as
    IntervalTrigger.start_date so every subsequent fire lands on the same
    UTC hour instead of drifting by whatever time the job was registered."""
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    if anchor <= now:
        anchor += timedelta(days=1)
    return anchor


def build_config_trigger(config) -> IntervalTrigger | CronTrigger:
    """Build the APScheduler trigger for a NewspaperConfig.

    Interval schedules honor run_hour too: we anchor the interval to the
    next occurrence of that UTC hour, so every fire lands on the hour the
    user picked instead of whatever time the job happened to be registered."""
    if config.schedule_kind == "weekdays":
        if not config.weekdays:
            raise ValueError("weekdays empty for schedule_kind='weekdays'")
        day_of_week = ",".join(_WEEKDAY_ABBR[d] for d in sorted(set(config.weekdays)))
        return CronTrigger(day_of_week=day_of_week, hour=config.run_hour, minute=0)

    if config.frequency_days is None:
        raise ValueError("frequency_days missing for schedule_kind='interval'")
    return IntervalTrigger(
        days=config.frequency_days,
        start_date=_next_utc_anchor(config.run_hour or 0),
    )


def register_config(config) -> None:
    """(Re-)register the recurring generation job for this config. Only
    active configs get a live trigger; inactive ones are removed so they
    don't fire in the background."""
    job_id = config_job_id(config.id)
    if not config.active:
        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass
        logger.info("scheduler: config %s inactive, no job registered", config.id)
        return

    try:
        trigger = build_config_trigger(config)
    except ValueError as exc:
        logger.warning("scheduler: cannot register config %s: %s", config.id, exc)
        return

    scheduler.add_job(
        "app.agent.runner:run_generation_job",
        trigger=trigger,
        args=[str(config.id)],
        id=job_id,
        replace_existing=True,
    )
    logger.info("scheduler: registered job %s (trigger=%s)", job_id, trigger)


def unregister_config(config_id) -> None:
    job_id = config_job_id(config_id)
    try:
        scheduler.remove_job(job_id)
        logger.info("scheduler: removed job %s", job_id)
    except Exception:
        logger.info("scheduler: no job to remove for %s", job_id)


def enqueue_manual_run_for_issue(issue_id) -> str:
    """Add a one-shot date-trigger job to run the agent against an already-
    created Issue row. The endpoint creates the Issue up front so the
    response can include its id; this job just executes the graph on it."""
    job_id = f"issue-{issue_id}-manual-{int(now_utc().timestamp())}"
    scheduler.add_job(
        "app.agent.runner:run_issue_job",
        trigger=DateTrigger(run_date=now_utc()),
        args=[str(issue_id)],
        id=job_id,
        replace_existing=False,
    )
    logger.info("scheduler: enqueued manual issue-run %s", job_id)
    return job_id
