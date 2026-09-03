import re
from typing import Tuple

INTENT_PATTERNS = {
    'weather': [
        r'weather', r'temperature', r'rain', r'sunny', r'forecast',
        r'how.*hot', r'how.*cold', r'climate',
    ],
    'news': [
        r'news', r'headlines', r'happening', r'latest', r'update',
        r'what.*world', r'current events',
    ],
    'music': [
        r'play', r'music', r'song', r'feeling', r'mood',
        r'relax', r'stressed', r'happy', r'sad', r'focus',
    ],
    'email': [
        r'email', r'mail', r'inbox', r'message from',
        r'check.*email', r'any.*emails', r'read.*email',
    ],
    'calendar': [
        r'calendar', r'schedule', r'meeting', r'appointment',
        r'what.*today', r'what.*tomorrow', r'upcoming',
    ],
    'memory': [
        r'remember', r'recall', r'what.*said', r'note',
        r'forget', r'store this', r'save this', r'recent memory',
    ],
    'tools': [
        r'run', r'execute', r'open', r'launch', r'start',
        r'search', r'calculate', r'compute', r'math',
    ],
    'greeting': [
        r'^hi$', r'^hello$', r'^hey', r'good morning',
        r'good evening', r'good afternoon', r'who are you',
    ],
    'exit': [
        r'exit', r'quit', r'shutdown', r'goodbye', r'bye',
        r'stop jarvis', r'turn off',
    ],
}

def classify_intent(text: str) -> Tuple[str, float]:
    text_lower = text.lower().strip()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent, 0.95
    return 'general', 0.5

def route(text: str) -> dict:
    intent, confidence = classify_intent(text)
    return {
        'text': text,
        'intent': intent,
        'confidence': confidence,
    }
