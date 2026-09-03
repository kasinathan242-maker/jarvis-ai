import httpx
from config.settings import settings

BASE_URL = 'https://newsapi.org/v2/top-headlines'

async def handle(text: str) -> str:
    if not settings.NEWS_API_KEY:
        return (
            "NewsAPI key is not set in .env. "
            "Top Headlines (Sample): 1. Next-Gen Autonomous AI Agents Released | "
            "2. Breakthroughs in Quantum Computing | 3. Global Tech Summit Announcements."
        )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(BASE_URL, params={
                'country': 'in',
                'pageSize': 5,
                'apiKey': settings.NEWS_API_KEY,
            })
            if response.status_code != 200:
                return f'Sorry, I could not fetch the latest news (HTTP {response.status_code}).'
            articles = response.json().get('articles', [])
            if not articles:
                return 'No news articles found at this time.'
            headlines = [f"{i+1}: {a['title']}" for i, a in enumerate(articles[:5])]
            return 'Here are the top headlines: ' + ' | '.join(headlines)
    except Exception as e:
        return f'Failed to retrieve news: {str(e)}'
