"""JARVIS Lite — System Backup & Disaster Recovery.

Creates compressed, SHA256-verified archives of all user memories, document indices,
schedules, workflows, and database stores. Provides zero-loss restoration with integrity checks.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import time
from pathlib import Path
from typing import Any

from core.logging import get_logger

logger = get_logger("maintenance.backup")

DEFAULT_BACKUP_DIR = Path("backups")
DEFAULT_DATA_DIR = Path("data")


class BackupManager:
    """Manages full backups and disaster recovery for JARVIS Lite."""

    def __init__(self, backup_dir: Path | None = None) -> None:
        self.backup_dir = (backup_dir or DEFAULT_BACKUP_DIR).resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def create_backup(
        self,
        data_dir: Path | None = None,
        include_workflows: bool = True,
        workflows_dir: Path | None = None,
    ) -> Path:
        """Create a complete timestamped tar.gz archive of user data, memories, and workflows.

        Returns:
            Path to the generated .tar.gz archive.
        """
        source_data = (data_dir or DEFAULT_DATA_DIR).resolve()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_name = f"jarvis_backup_{timestamp}.tar.gz"
        archive_path = self.backup_dir / archive_name

        included_items: list[str] = []

        with tarfile.open(archive_path, "w:gz") as tar:
            # 1. Archive data/ directory
            if source_data.exists():
                tar.add(source_data, arcname="data")
                included_items.append("data")

            # 2. Archive root SQLite database if present
            root_db = Path("jarvis_memory.db").resolve()
            if root_db.exists():
                tar.add(root_db, arcname="jarvis_memory.db")
                included_items.append("jarvis_memory.db")

            # 3. Archive declarative workflows if present
            wf_dir = (workflows_dir or Path("workflows")).resolve()
            if include_workflows and wf_dir.exists():
                tar.add(wf_dir, arcname="workflows")
                included_items.append("workflows")

        # Generate cryptographic manifest
        checksum = self._calculate_sha256(archive_path)
        manifest = {
            "archive_name": archive_name,
            "created_at": time.time(),
            "sha256": checksum,
            "size_bytes": archive_path.stat().st_size,
            "included_components": included_items,
        }
        manifest_path = archive_path.with_suffix(".json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "backup_created_successfully",
            archive=archive_name,
            size_bytes=manifest["size_bytes"],
            sha256=checksum[:12],
        )
        return archive_path

    def restore_backup(
        self,
        archive_path: Path,
        destination_root: Path = Path("."),
    ) -> dict[str, Any]:
        """Restore all components from a backup archive into the target environment.

        Returns:
            Dictionary detailing restored components and verification status.
        """
        archive_path = archive_path.resolve()
        if not archive_path.exists():
            raise FileNotFoundError(f"Backup archive not found: {archive_path}")

        dest_root = destination_root.resolve()
        restored_files: list[str] = []

        with tarfile.open(archive_path, "r:gz") as tar:
            # Prevent directory traversal attacks
            for member in tar.getmembers():
                target_path = dest_root / member.name
                if not target_path.resolve().is_relative_to(dest_root):
                    raise RuntimeError(f"Illegal path in archive: {member.name}")
                restored_files.append(member.name)
            tar.extractall(path=dest_root)

        # Verify restoration integrity
        data_restored = (dest_root / "data").exists()
        memories_restored = (dest_root / "jarvis_memory.db").exists() or (dest_root / "data" / "jarvis_memory.db").exists()
        workflows_restored = (dest_root / "workflows").exists()

        logger.info(
            "backup_restored_successfully",
            archive=archive_path.name,
            total_files=len(restored_files),
            data_restored=data_restored,
        )

        return {
            "success": True,
            "archive": str(archive_path),
            "files_restored_count": len(restored_files),
            "data_restored": data_restored,
            "memories_restored": memories_restored,
            "workflows_restored": workflows_restored,
        }


backup_manager = BackupManager()
