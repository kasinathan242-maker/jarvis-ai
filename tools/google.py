"""JARVIS Lite — Google Services Tools (Gmail & Calendar).

Exposes email search, reply drafting, and calendar management.
Enforces Rule: SENDING AN EMAIL ALWAYS REQUIRES EXPLICIT CONFIRMATION.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from integrations.google import google_service
from security.permissions import RiskLevel
from tools.base import BaseTool


class SearchEmailsInput(BaseModel):
    query: str = Field(default="", description="Search query string (e.g. sender name or subject keyword)")
    max_results: int = Field(default=5, ge=1, le=20)


class EmailSummaryItem(BaseModel):
    id: str
    sender: str
    subject: str
    snippet: str
    date: str


class SearchEmailsOutput(BaseModel):
    count: int
    emails: list[EmailSummaryItem]
    verified: bool


class SearchEmailsTool(BaseTool[SearchEmailsInput, SearchEmailsOutput]):
    name = "search_emails"
    description = "Search emails in Gmail inbox by keyword, subject, or sender."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 15.0
    input_schema = SearchEmailsInput
    output_schema = SearchEmailsOutput

    async def _execute(self, params: SearchEmailsInput) -> SearchEmailsOutput:
        items = google_service.search_messages(query=params.query, max_results=params.max_results)
        summaries = [
            EmailSummaryItem(
                id=item["id"],
                sender=item["from"],
                subject=item["subject"],
                snippet=item["body"][:120],
                date=item.get("date", ""),
            )
            for item in items
        ]
        return SearchEmailsOutput(
            count=len(summaries),
            emails=summaries,
            verified=True,
        )

    async def verify(self, params: SearchEmailsInput, result: SearchEmailsOutput) -> bool:
        result.verified = True
        return True


class DraftEmailReplyInput(BaseModel):
    to_email: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    reply_body: str = Field(description="Draft body text for the email reply")
    thread_id: str | None = Field(default=None, description="Optional thread identifier")


class DraftEmailReplyOutput(BaseModel):
    draft_id: str
    to_email: str
    subject: str
    preview_body: str
    created_at: str
    requires_approval_to_send: bool
    verified: bool


class DraftEmailReplyTool(BaseTool[DraftEmailReplyInput, DraftEmailReplyOutput]):
    name = "draft_email_reply"
    description = "Create a draft email reply. Does NOT send the email until separate user approval is granted."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 15.0
    input_schema = DraftEmailReplyInput
    output_schema = DraftEmailReplyOutput

    async def _execute(self, params: DraftEmailReplyInput) -> DraftEmailReplyOutput:
        draft = google_service.draft_reply(
            to_email=params.to_email,
            subject=params.subject,
            body=params.reply_body,
            thread_id=params.thread_id,
        )
        return DraftEmailReplyOutput(
            draft_id=draft["draft_id"],
            to_email=draft["to"],
            subject=draft["subject"],
            preview_body=draft["body"],
            created_at=draft["created_at"],
            requires_approval_to_send=True,
            verified=True,
        )

    async def verify(self, params: DraftEmailReplyInput, result: DraftEmailReplyOutput) -> bool:
        result.verified = bool(result.draft_id.startswith("draft_"))
        return result.verified


class SendEmailInput(BaseModel):
    to_email: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Body content to send")


class SendEmailOutput(BaseModel):
    sent_id: str
    to_email: str
    subject: str
    sent_at: str
    verified: bool


class SendEmailTool(BaseTool[SendEmailInput, SendEmailOutput]):
    name = "send_email"
    description = "Send an email message via Gmail. STRICTLY GATED by user confirmation approval ticket."
    risk_level = RiskLevel.CONFIRM  # ALWAYS REQUIRES EXPLICIT APPROVAL
    timeout_seconds = 20.0
    input_schema = SendEmailInput
    output_schema = SendEmailOutput

    async def _execute(self, params: SendEmailInput) -> SendEmailOutput:
        sent = google_service.send_message(
            to_email=params.to_email,
            subject=params.subject,
            body=params.body,
        )
        return SendEmailOutput(
            sent_id=sent["id"],
            to_email=sent["to"],
            subject=sent["subject"],
            sent_at=sent["sent_at"],
            verified=True,
        )

    async def verify(self, params: SendEmailInput, result: SendEmailOutput) -> bool:
        result.verified = bool(result.sent_id.startswith("sent_"))
        return result.verified


class ListCalendarEventsInput(BaseModel):
    max_results: int = Field(default=5, ge=1, le=20)


class CalendarEventItem(BaseModel):
    id: str
    summary: str
    start: str
    end: str
    description: str


class ListCalendarEventsOutput(BaseModel):
    count: int
    events: list[CalendarEventItem]
    verified: bool


class ListCalendarEventsTool(BaseTool[ListCalendarEventsInput, ListCalendarEventsOutput]):
    name = "list_calendar_events"
    description = "List upcoming events from Google Calendar."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 15.0
    input_schema = ListCalendarEventsInput
    output_schema = ListCalendarEventsOutput

    async def _execute(self, params: ListCalendarEventsInput) -> ListCalendarEventsOutput:
        raw_events = google_service.list_calendar_events(max_results=params.max_results)
        items = [
            CalendarEventItem(
                id=e["id"],
                summary=e["summary"],
                start=e["start"],
                end=e["end"],
                description=e.get("description", ""),
            )
            for e in raw_events
        ]
        return ListCalendarEventsOutput(
            count=len(items),
            events=items,
            verified=True,
        )

    async def verify(self, params: ListCalendarEventsInput, result: ListCalendarEventsOutput) -> bool:
        result.verified = True
        return True


class CreateCalendarEventInput(BaseModel):
    summary: str = Field(description="Event title or summary")
    start_iso: str = Field(description="Start time in ISO format (e.g. 2026-10-22T14:00:00Z)")
    end_iso: str = Field(description="End time in ISO format (e.g. 2026-10-22T15:00:00Z)")
    description: str | None = Field(default=None, description="Optional description")


class CreateCalendarEventOutput(BaseModel):
    event_id: str
    summary: str
    start_iso: str
    end_iso: str
    verified: bool


class CreateCalendarEventTool(BaseTool[CreateCalendarEventInput, CreateCalendarEventOutput]):
    name = "create_calendar_event"
    description = "Schedule a new event in Google Calendar."
    risk_level = RiskLevel.CONFIRM
    timeout_seconds = 20.0
    input_schema = CreateCalendarEventInput
    output_schema = CreateCalendarEventOutput

    async def _execute(self, params: CreateCalendarEventInput) -> CreateCalendarEventOutput:
        evt = google_service.create_calendar_event(
            summary=params.summary,
            start_iso=params.start_iso,
            end_iso=params.end_iso,
            description=params.description,
        )
        return CreateCalendarEventOutput(
            event_id=evt["id"],
            summary=evt["summary"],
            start_iso=evt["start"],
            end_iso=evt["end"],
            verified=True,
        )

    async def verify(self, params: CreateCalendarEventInput, result: CreateCalendarEventOutput) -> bool:
        result.verified = bool(result.event_id.startswith("evt_"))
        return result.verified
