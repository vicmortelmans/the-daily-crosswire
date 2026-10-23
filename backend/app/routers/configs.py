import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.db import get_session
from app.models import NewspaperConfig, Topic
from app.schemas import (
    ConfigCreate,
    ConfigListItem,
    ConfigRead,
    ConfigUpdate,
    validate_schedule,
)
from app.scheduler import enqueue_manual_run_for_issue, register_config, unregister_config
from app.utils.time import now_utc

router = APIRouter(prefix="/api/configs", tags=["configs"])


@router.get("", response_model=list[ConfigListItem])
def list_configs(session: Session = Depends(get_session)):
    return session.exec(select(NewspaperConfig).order_by(NewspaperConfig.created_at.desc())).all()


@router.post("", response_model=ConfigRead, status_code=status.HTTP_201_CREATED)
def create_config(payload: ConfigCreate, session: Session = Depends(get_session)):
    data = payload.model_dump()
    # Zero out the orphan field for the chosen mode so the DB never holds both.
    if data["schedule_kind"] == "interval":
        data["weekdays"] = None
    else:
        data["frequency_days"] = None

    config = NewspaperConfig(**data)
    session.add(config)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="name already in use")
    session.refresh(config)
    register_config(config)
    return _load_with_topics(session, config.id)


@router.get("/{config_id}", response_model=ConfigRead)
def get_config(config_id: uuid.UUID, session: Session = Depends(get_session)):
    config = _load_with_topics(session, config_id)
    if config is None:
        raise HTTPException(status_code=404, detail="config not found")
    return config


@router.patch("/{config_id}", response_model=ConfigRead)
def update_config(
    config_id: uuid.UUID,
    payload: ConfigUpdate,
    session: Session = Depends(get_session),
):
    config = session.get(NewspaperConfig, config_id)
    if config is None:
        raise HTTPException(status_code=404, detail="config not found")

    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        # This switch is non-nullable in the DB; ignore an explicit null.
        if key == "include_workflow_report" and value is None:
            continue
        setattr(config, key, value)

    # Validate the post-merge state and clean up the orphan schedule field.
    try:
        validate_schedule(config.schedule_kind, config.frequency_days, config.weekdays)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if config.schedule_kind == "interval":
        config.weekdays = None
    else:
        config.frequency_days = None

    config.updated_at = now_utc()

    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="name already in use")
    session.refresh(config)
    register_config(config)
    return _load_with_topics(session, config.id)


@router.delete("/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_config(config_id: uuid.UUID, session: Session = Depends(get_session)):
    config = session.get(NewspaperConfig, config_id)
    if config is None:
        raise HTTPException(status_code=404, detail="config not found")
    session.delete(config)
    session.commit()
    unregister_config(config_id)
    return None


@router.post("/{config_id}/generate", status_code=status.HTTP_202_ACCEPTED)
def trigger_generate(config_id: uuid.UUID, session: Session = Depends(get_session)):
    """Create a pending Issue synchronously, enqueue the agent job against
    that Issue, and return its id so the UI can poll /api/issues/{id} without
    guessing which row the scheduler just inserted."""
    from app.agent.runner import create_pending_issue

    config = session.get(NewspaperConfig, config_id)
    if config is None:
        raise HTTPException(status_code=404, detail="config not found")
    try:
        issue_id = create_pending_issue(config_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    job_id = enqueue_manual_run_for_issue(issue_id)
    return {
        "job_id": job_id,
        "issue_id": str(issue_id),
        "config_id": str(config_id),
        "status": "queued",
    }


def _load_with_topics(session: Session, config_id: uuid.UUID) -> NewspaperConfig | None:
    return session.exec(
        select(NewspaperConfig)
        .where(NewspaperConfig.id == config_id)
        .options(selectinload(NewspaperConfig.topics).selectinload(Topic.feeds))
    ).first()
