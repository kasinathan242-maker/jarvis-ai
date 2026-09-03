from .weather import handle as handle_weather
from .news import handle as handle_news
from .music import handle as handle_music
from .email_handler import handle as handle_email
from .calendar_handler import handle as handle_calendar
from .tools import handle as handle_tools

__all__ = [
    "handle_weather",
    "handle_news",
    "handle_music",
    "handle_email",
    "handle_calendar",
    "handle_tools",
]
