from datetime import datetime
import re

def format_timestamp(dt: datetime = None) -> str:
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def clean_string(text: str) -> str:
    if not text:
        return ""
    # Strip HTML tags or invalid non-printable characters
    cleaned = re.sub(r'<[^>]+>', '', text)
    return cleaned.strip()
