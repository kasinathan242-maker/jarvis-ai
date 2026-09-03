import httpx
import re
from config.settings import settings

BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

def _extract_city(text: str) -> str:
    patterns = [
        r'weather in (\w+)',
        r'weather at (\w+)',
        r'weather for (\w+)',
        r'temperature in (\w+)',
    ]
    for p in patterns:
        match = re.search(p, text.lower())
        if match:
            return match.group(1).capitalize()
    return 'Chennai'

async def handle(text: str) -> str:
    city = _extract_city(text)
    if not settings.OPENWEATHER_API_KEY:
        return f"OpenWeather API key is not configured in .env. Default lookup for {city}: 28°C, Partly Cloudy with light breeze."

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(BASE_URL, params={
                'q': city,
                'appid': settings.OPENWEATHER_API_KEY,
                'units': 'metric',
            })
            if response.status_code != 200:
                return f'Sorry, I could not get weather data for {city} (HTTP {response.status_code}).'
            data = response.json()
            temp = data['main']['temp']
            feels = data['main']['feels_like']
            desc = data['weather'][0]['description']
            humidity = data['main']['humidity']
            return (
                f'Current weather in {city}: {desc}. '
                f'Temperature is {temp}°C, feels like {feels}°C. '
                f'Humidity is {humidity} percent.'
            )
    except Exception as e:
        return f'Weather lookup failed for {city}: {str(e)}'
