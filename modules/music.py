import webbrowser

MOOD_QUERIES = {
    'stressed': 'relaxing calm music playlist',
    'sad': 'uplifting happy music playlist',
    'happy': 'happy feel good music playlist',
    'focus': 'focus deep work music playlist',
    'workout': 'high energy workout music playlist',
    'sleep': 'sleep relaxing ambient music playlist',
    'morning': 'morning positive energy music playlist',
    'relax': 'relaxing lofi chill music playlist',
}

KEYWORD_MAP = {
    'stress': 'stressed', 'stressed': 'stressed', 'anxious': 'stressed',
    'sad': 'sad', 'depressed': 'sad', 'down': 'sad',
    'happy': 'happy', 'excited': 'happy', 'good': 'happy',
    'focus': 'focus', 'study': 'focus', 'work': 'focus', 'concentrate': 'focus',
    'workout': 'workout', 'gym': 'workout', 'exercise': 'workout',
    'sleep': 'sleep', 'tired': 'sleep', 'rest': 'sleep',
    'morning': 'morning', 'wake': 'morning',
    'relax': 'relax', 'chill': 'relax', 'calm': 'relax',
}

def _detect_mood(text: str) -> str:
    text_lower = text.lower()
    for keyword, mood in KEYWORD_MAP.items():
        if keyword in text_lower:
            return mood
    return 'relax'

async def handle(text: str) -> str:
    mood = _detect_mood(text)
    query = MOOD_QUERIES.get(mood, MOOD_QUERIES['relax'])
    url = f'https://www.youtube.com/results?search_query={query.replace(" ", "+")}'
    try:
        webbrowser.open(url)
    except Exception:
        pass
    return f'Playing {mood} music for you. Enjoy! URL: {url}'
