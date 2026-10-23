import uuid
from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class FeedRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    topic_id: uuid.UUID
    url: str
    last_polled_at: Optional[datetime] = None


class FeedCreate(BaseModel):
    url: str = Field(min_length=1)


class TopicRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    config_id: uuid.UUID
    name: str
    preferred_sources: list[str] = []
    search_hints: Optional[str] = None
    feeds: list[FeedRead] = []


class TopicCreate(BaseModel):
    name: str = Field(min_length=1)
    preferred_sources: list[str] = []
    search_hints: Optional[str] = None


class TopicUpdate(BaseModel):
    name: Optional[str] = None
    preferred_sources: Optional[list[str]] = None
    search_hints: Optional[str] = None


ScheduleKind = Literal["interval", "weekdays"]

LanguageCode = Literal["en", "es", "fr", "it", "pt", "de", "nl"]


def validate_schedule(
    schedule_kind: str,
    frequency_days: Optional[int],
    weekdays: Optional[list[int]],
) -> None:
    if schedule_kind == "interval":
        if frequency_days is None:
            raise ValueError("frequency_days required when schedule_kind='interval'")
    elif schedule_kind == "weekdays":
        if not weekdays:
            raise ValueError("weekdays must be non-empty when schedule_kind='weekdays'")
        if any(d < 0 or d > 6 for d in weekdays):
            raise ValueError("weekdays entries must be 0-6 (0=Mon..6=Sun)")
    else:
        raise ValueError("schedule_kind must be 'interval' or 'weekdays'")


class ConfigRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    schedule_kind: ScheduleKind
    frequency_days: Optional[int] = None
    weekdays: Optional[list[int]] = None
    run_hour: int
    include_puzzles: bool
    active: bool
    max_stories_per_topic: Optional[int] = None
    language: LanguageCode = "en"
    include_workflow_report: bool = True
    newsletter_emails: list[EmailStr] = []
    created_at: datetime
    updated_at: datetime
    topics: list[TopicRead] = []


class ConfigListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    schedule_kind: ScheduleKind
    frequency_days: Optional[int] = None
    weekdays: Optional[list[int]] = None
    run_hour: int
    include_puzzles: bool
    active: bool
    max_stories_per_topic: Optional[int] = None
    language: LanguageCode = "en"
    include_workflow_report: bool = True
    created_at: datetime
    updated_at: datetime


class ConfigCreate(BaseModel):
    name: str = Field(min_length=1)
    schedule_kind: ScheduleKind = "interval"
    frequency_days: Optional[int] = Field(default=None, ge=1, le=365)
    weekdays: Optional[list[int]] = None
    run_hour: int = Field(default=6, ge=0, le=23)
    include_puzzles: bool = True
    active: bool = True
    max_stories_per_topic: Optional[int] = Field(default=None, ge=1, le=50)
    language: LanguageCode = "en"
    include_workflow_report: bool = True
    newsletter_emails: list[EmailStr] = []

    @model_validator(mode="after")
    def _check_schedule(self):
        validate_schedule(self.schedule_kind, self.frequency_days, self.weekdays)
        return self


class ConfigUpdate(BaseModel):
    name: Optional[str] = None
    schedule_kind: Optional[ScheduleKind] = None
    frequency_days: Optional[int] = Field(default=None, ge=1, le=365)
    weekdays: Optional[list[int]] = None
    run_hour: Optional[int] = Field(default=None, ge=0, le=23)
    include_puzzles: Optional[bool] = None
    active: Optional[bool] = None
    max_stories_per_topic: Optional[int] = Field(default=None, ge=1, le=50)
    language: Optional[LanguageCode] = None
    include_workflow_report: Optional[bool] = None
    newsletter_emails: Optional[list[EmailStr]] = None


class IssueSendRequest(BaseModel):
    """Ad-hoc send of a ready issue to a one-off recipient list. Distinct
    from the per-config newsletter which fires automatically at notify
    time."""

    emails: list[EmailStr] = Field(min_length=1)


class RecipientOutcome(BaseModel):
    email: str
    status: Literal["ok", "failed"]
    error: Optional[str] = None


class IssueSendResponse(BaseModel):
    issue_id: uuid.UUID
    sent: int
    failed: int
    per_recipient: list[RecipientOutcome]


class IssueListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    config_id: uuid.UUID
    config_name: Optional[str] = None
    slug: str
    issue_date: date
    lookback_start: Optional[date] = None
    lookback_end: Optional[date] = None
    status: str
    error: Optional[str] = None
    html_path: Optional[str] = None
    created_at: datetime


class IssueDetail(IssueListItem):
    drafts_count: int = 0
    has_editorial: bool = False
    has_puzzles: bool = False
    warnings: list[str] = []
