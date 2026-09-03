import asyncio
from core.intent_router import route
from voice.speaker import speak
from memory.store import save_conversation
from memory.recall import get_recent_conversations
from memory.db import init_db

import modules.weather as weather_mod
import modules.news as news_mod
import modules.music as music_mod
import modules.email_handler as email_mod
import modules.calendar_handler as calendar_mod
import modules.tools as tools_mod

MODULE_MAP = {
    'weather': weather_mod.handle,
    'news': news_mod.handle,
    'music': music_mod.handle,
    'email': email_mod.handle,
    'calendar': calendar_mod.handle,
    'tools': tools_mod.handle,
}

class Jarvis:
    def __init__(self):
        init_db()
        print('[JARVIS] System Initialized & Ready.')

    async def handle_command(self, text: str) -> str:
        if not text or len(text.strip()) < 2:
            return 'Could not understand command. Please try again.'

        routed = route(text)
        intent = routed['intent']
        confidence = routed['confidence']
        print(f'[INTENT DETECTED]: "{intent}" (confidence={confidence})')

        if intent == 'exit':
            response = 'Goodbye! Shutting down JARVIS systems.'
            await speak(response)
            return '__EXIT__'

        if intent == 'greeting':
            response = 'Hello! I am JARVIS, your voice-driven autonomous desktop assistant. How can I help you today?'
        elif intent == 'memory':
            history = get_recent_conversations(3)
            if history:
                items = [f'"{h["user_input"]}" -> Intent: {h["intent"]}' for h in history]
                response = 'Recent conversation context from memory: ' + ' | '.join(items)
            else:
                response = 'No memory stored yet. Say a command to save memory context.'
        elif intent in MODULE_MAP:
            handler = MODULE_MAP[intent]
            response = await handler(text)
        else:
            # Fallback general query handling
            response = (
                f'Command received: "{text}". Handled under general intent. '
                'You can ask for weather, news, music, emails, calendar events, calculations, web searches, or opening apps.'
            )

        save_conversation(text, intent, response)
        await speak(response)
        return response

    async def run_loop(self, transcribe_fn) -> None:
        print('[JARVIS] Starting autonomous voice listener loop...')
        await speak('JARVIS online. Autonomous voice systems active.')

        while True:
            try:
                text = transcribe_fn()
                if not text:
                    await asyncio.sleep(1)
                    continue

                result = await self.handle_command(text)
                if result == '__EXIT__':
                    break
            except KeyboardInterrupt:
                print('\n[JARVIS] Session terminated by user.')
                break
            except Exception as e:
                print(f'[JARVIS Error]: {e}')
                await asyncio.sleep(1)
