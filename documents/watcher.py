"""JARVIS Lite — Automated Folder Watcher for Document Indexing.

Monitors watched directories using watchdog. When a new or modified document
(PDF, DOCX, TXT, MD) is dropped in, automatically indexes it into ChromaDB
within 30 seconds, applying strict folder exclusion lists.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from core.event_bus import bus
from core.logging import get_logger
from documents.indexer import document_indexer

logger = get_logger("documents.watcher")

DEFAULT_WATCH_DIR = Path("data/watched_docs")

EXCLUDE_PATTERNS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "AppData",
    "Windows",
    "System32",
    ".temp",
    ".tmp",
}

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


class DocumentDropHandler(FileSystemEventHandler):
    """File system event handler that notifies indexer of new/modified documents."""

    def __init__(self, on_file_event: Callable[[Path], None]) -> None:
        super().__init__()
        self._on_file_event = on_file_event

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_path(Path(str(event.src_path)))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_path(Path(str(event.src_path)))

    def _handle_path(self, path: Path) -> None:
        # Check exclusion rules
        for part in path.parts:
            if part in EXCLUDE_PATTERNS or part.startswith("."):
                return

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            self._on_file_event(path)


class DocumentWatcher:
    """Watches a directory and auto-indexes documents into ChromaDB within 30 seconds."""

    def __init__(self, watch_dir: Path | None = None) -> None:
        self.watch_dir = (watch_dir or DEFAULT_WATCH_DIR).resolve()
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self._observer: BaseObserver | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending_files: dict[Path, float] = {}
        self._lock = threading.Lock()
        self._is_running = False

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the background watchdog observer thread."""
        if self._is_running:
            return

        try:
            self._loop = loop or asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()

        handler = DocumentDropHandler(self._queue_file_for_indexing)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.watch_dir), recursive=True)
        self._observer.start()
        self._is_running = True
        logger.info("document_watcher_started", watch_dir=str(self.watch_dir))

    def _queue_file_for_indexing(self, path: Path) -> None:
        with self._lock:
            self._pending_files[path] = time.time()

        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._process_single_file(path), self._loop)

    async def _process_single_file(self, path: Path) -> None:
        """Index a single document file, handling file-locking delays."""
        # Wait a short moment to ensure file write stream is complete
        await asyncio.sleep(0.5)
        if not path.exists():
            return

        try:
            count = await document_indexer.index_file(path, project="watched_docs")
            logger.info("auto_indexed_watched_document", file_name=path.name, chunks=count)
            await bus.publish(
                "document_auto_indexed",
                {
                    "file_name": path.name,
                    "file_path": str(path),
                    "chunks": count,
                    "timestamp": time.time(),
                },
            )
        except Exception as exc:
            logger.error("auto_index_failed", file_name=path.name, error=str(exc))

    def stop(self) -> None:
        """Stop watcher observer."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=3.0)
            self._observer = None
        self._is_running = False
        logger.info("document_watcher_stopped")


document_watcher = DocumentWatcher()
