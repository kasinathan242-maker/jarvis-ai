import asyncio
from datetime import datetime, timedelta, timezone
import os
from config.settings import settings

SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
TOKEN_PATH = 'credentials/token.json'

def _fetch_events_sync() -> list:
    if not os.path.exists(settings.GOOGLE_CREDENTIALS_PATH) and not os.path.exists(TOKEN_PATH):
        return []

    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif os.path.exists(settings.GOOGLE_CREDENTIALS_PATH):
            flow = InstalledAppFlow.from_client_secrets_file(
                settings.GOOGLE_CREDENTIALS_PATH, SCOPES
            )
            creds = flow.run_local_server(port=0)
            os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
            with open(TOKEN_PATH, 'w') as f:
                f.write(creds.to_json())
        else:
            return []

    service = build('calendar', 'v3', credentials=creds)
    now = datetime.now(timezone.utc).isoformat()
    end = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    result = service.events().list(
        calendarId='primary',
        timeMin=now,
        timeMax=end,
        maxResults=5,
        singleEvents=True,
        orderBy='startTime',
    ).execute()
    return result.get('items', [])

async def handle(text: str) -> str:
    if not os.path.exists(settings.GOOGLE_CREDENTIALS_PATH) and not os.path.exists(TOKEN_PATH):
        return "Google Calendar credentials file not found. Place credentials in credentials/google_credentials.json to enable calendar sync."

    try:
        loop = asyncio.get_running_loop()
        events = await loop.run_in_executor(None, _fetch_events_sync)
    except Exception as e:
        return f'Could not fetch calendar events: {str(e)}'

    if not events:
        return 'You have no upcoming events today.'

    lines = []
    for e in events:
        start = e['start'].get('dateTime', e['start'].get('date', ''))
        lines.append(f"{e.get('summary', 'Event')} at {start}")
    return 'Your upcoming events: ' + ', '.join(lines)
