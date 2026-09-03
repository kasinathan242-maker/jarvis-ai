import asyncio
import logging
from typing import Callable, Any

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

class AsyncEngine:
    """Async event loop task runner and worker manager for JARVIS AI."""

    def __init__(self):
        self.logger = logging.getLogger("JarvisAsyncEngine")

    async def execute_task(self, func: Callable, *args, **kwargs) -> Any:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

async_engine = AsyncEngine()
