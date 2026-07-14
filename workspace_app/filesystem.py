from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .config import Settings


class WorkspacePathError(ValueError):
    pass


class WorkspaceLimitError(ValueError):
    pass


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    files: Path
    artifacts: Path
    metadata: Path

    def run_artifacts(self, run_id: str) -> Path:
        return self.artifacts / "runs" / run_id

    def run_base(self, run_id: str) -> Path:
        return self.run_artifacts(run_id) / "base"

    def run_staging(self, run_id: str) -> Path:
        return self.run_artifacts(run_id) / "staging"


class WorkspaceFilesystem:
    def __init__(self, settings: Settings):
        self.settings = settings

    def paths(self, workspace_id: str) -> WorkspacePaths:
        if not workspace_id or any(char not in "0123456789abcdef-" for char in workspace_id.casefold()):
            raise WorkspacePathError("Invalid workspace id.")
        root = (self.settings.workspaces_root / workspace_id).resolve()
        try:
            root.relative_to(self.settings.workspaces_root.resolve())
        except ValueError as exc:
            raise WorkspacePathError("Workspace path escapes the data root.") from exc
        return WorkspacePaths(root=root, files=root / "files", artifacts=root / "artifacts", metadata=root / "workspace.json")

    def create_workspace(self, workspace_id: str, metadata: dict[str, Any]) -> WorkspacePaths:
        paths = self.paths(workspace_id)
        paths.files.mkdir(parents=True, exist_ok=False)
        paths.artifacts.mkdir(parents=True, exist_ok=True)
        paths.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return paths

    def update_metadata(self, workspace_id: str, metadata: dict[str, Any]) -> None:
        paths = self.paths(workspace_id)
        paths.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete_workspace(self, workspace_id: str) -> None:
        paths = self.paths(workspace_id)
        if paths.root.exists():
            shutil.rmtree(paths.root)

    def resolve(self, root: Path, requested_path: str, *, allow_root: bool = False, allow_missing: bool = False) -> Path:
        if not isinstance(requested_path, str) or "\x00" in requested_path:
            raise WorkspacePathError("Invalid path.")
        normalized = requested_path.strip().replace("\\", "/")
        if normalized in {"", "."}:
            if allow_root:
                return root.resolve()
            raise WorkspacePathError("A file path is required.")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts:
            raise WorkspacePathError("Path is outside the workspace.")
        root_resolved = root.resolve()
        candidate = root_resolved.joinpath(*pure.parts)
        current = root_resolved
        for part in pure.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise WorkspacePathError("Symbolic links are not allowed.")
        resolved = candidate.resolve(strict=not allow_missing)
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise WorkspacePathError("Path is outside the workspace.") from exc
        return resolved

    def _regular_text_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(str(path))
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise WorkspacePathError("Only regular files are supported.")
        if path.stat().st_size > self.settings.max_file_bytes:
            raise WorkspaceLimitError("File exceeds the 1 MB MVP limit.")

    def list_files(self, root: Path, requested_path: str = ".") -> list[dict[str, Any]]:
        directory = self.resolve(root, requested_path, allow_root=True)
        if not directory.is_dir():
            raise WorkspacePathError("Path is not a directory.")
        result: list[dict[str, Any]] = []
        for item in sorted(directory.rglob("*"), key=lambda value: value.as_posix().casefold()):
            if item.is_symlink():
                raise WorkspacePathError("Symbolic links are not allowed.")
            relative = item.relative_to(root.resolve()).as_posix()
            result.append({
                "path": relative,
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
                "modifiedAt": datetime.fromtimestamp(item.stat().st_mtime, timezone.utc).isoformat(),
            })
        return result

    def read_file(self, root: Path, requested_path: str) -> str:
        path = self.resolve(root, requested_path)
        self._regular_text_file(path)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspacePathError("Only UTF-8 text files are supported.") from exc

    def write_file(self, root: Path, requested_path: str, content: str) -> Path:
        encoded = content.encode("utf-8")
        if len(encoded) > self.settings.max_file_bytes:
            raise WorkspaceLimitError("File exceeds the 1 MB MVP limit.")
        path = self.resolve(root, requested_path, allow_missing=True)
        parent = self.resolve(root, path.parent.relative_to(root.resolve()).as_posix(), allow_root=True, allow_missing=True)
        parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise WorkspacePathError("Only regular files are supported.")
        path.write_text(content, encoding="utf-8")
        self.validate_tree(root)
        return path

    def search_files(self, root: Path, query: str, requested_path: str = ".", glob: str | None = None) -> list[dict[str, Any]]:
        if not query:
            raise WorkspacePathError("Search query cannot be empty.")
        directory = self.resolve(root, requested_path, allow_root=True)
        pattern = glob or "*"
        if PurePosixPath(pattern).is_absolute() or ".." in PurePosixPath(pattern).parts or "/" in pattern or "\\" in pattern:
            raise WorkspacePathError("Search glob must be a file-name pattern such as *.py.")
        matches: list[dict[str, Any]] = []
        needle = query.casefold()
        for path in directory.rglob(pattern):
            if len(matches) >= 200:
                break
            if path.is_symlink():
                raise WorkspacePathError("Symbolic links are not allowed.")
            if not path.is_file() or path.stat().st_size > self.settings.max_file_bytes:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if needle in line.casefold():
                    matches.append({"path": path.relative_to(root.resolve()).as_posix(), "line": line_number, "text": line[:500]})
                    if len(matches) >= 200:
                        break
        return matches

    def validate_tree(self, root: Path) -> int:
        total = 0
        if not root.exists():
            return total
        for path in root.rglob("*"):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise WorkspacePathError(f"Symbolic links are not allowed: {path.name}")
            if path.is_file():
                if not stat.S_ISREG(mode):
                    raise WorkspacePathError("Only regular files are supported.")
                size = path.stat().st_size
                if size > self.settings.max_file_bytes:
                    raise WorkspaceLimitError(f"File exceeds limit: {path.name}")
                total += size
                if total > self.settings.max_workspace_bytes:
                    raise WorkspaceLimitError("Workspace exceeds the 50 MB MVP limit.")
        return total

    def prepare_run(self, workspace_id: str, run_id: str) -> tuple[Path, Path]:
        paths = self.paths(workspace_id)
        run_root = paths.run_artifacts(run_id)
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copytree(paths.files, paths.run_base(run_id), symlinks=True)
        shutil.copytree(paths.files, paths.run_staging(run_id), symlinks=True)
        self.validate_tree(paths.run_base(run_id))
        self.validate_tree(paths.run_staging(run_id))
        return paths.run_base(run_id), paths.run_staging(run_id)

    def host_staging_path(self, workspace_id: str, run_id: str) -> Path:
        relative = self.paths(workspace_id).run_staging(run_id).relative_to(self.settings.data_root)
        return self.settings.host_data_root / relative

    @staticmethod
    def _sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def manifest(self, root: Path) -> dict[str, str]:
        self.validate_tree(root)
        return {
            path.relative_to(root).as_posix(): self._sha(path)
            for path in root.rglob("*")
            if path.is_file()
        }

    def diff(self, base: Path, staging: Path) -> list[dict[str, Any]]:
        before = self.manifest(base)
        after = self.manifest(staging)
        changes: list[dict[str, Any]] = []
        for relative in sorted(set(before) | set(after)):
            if before.get(relative) == after.get(relative):
                continue
            if relative not in before:
                change_type = "created"
            elif relative not in after:
                change_type = "deleted"
            else:
                change_type = "modified"
            old_lines = self._safe_lines(base / relative) if relative in before else []
            new_lines = self._safe_lines(staging / relative) if relative in after else []
            patch = "".join(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{relative}", tofile=f"b/{relative}"))
            changes.append({
                "path": relative,
                "changeType": change_type,
                "diff": patch,
                "beforeSha256": before.get(relative),
                "afterSha256": after.get(relative),
            })
        return changes

    def _safe_lines(self, path: Path) -> list[str]:
        try:
            return path.read_text(encoding="utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            return ["Binary file omitted from textual diff.\n"]

    def apply_staging(self, workspace_id: str, run_id: str) -> None:
        paths = self.paths(workspace_id)
        staging = paths.run_staging(run_id)
        self.validate_tree(staging)
        replacement = paths.root / f"files.apply-{run_id}"
        backup = paths.root / f"files.backup-{run_id}"
        if replacement.exists():
            shutil.rmtree(replacement)
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(staging, replacement)
        self.validate_tree(replacement)
        os.replace(paths.files, backup)
        try:
            os.replace(replacement, paths.files)
        except Exception:
            os.replace(backup, paths.files)
            raise
        shutil.rmtree(backup)

    def discard_staging(self, workspace_id: str, run_id: str) -> None:
        staging = self.paths(workspace_id).run_staging(run_id)
        if staging.exists():
            shutil.rmtree(staging)

    def remove_run_base(self, workspace_id: str, run_id: str) -> None:
        base = self.paths(workspace_id).run_base(run_id)
        if base.exists():
            shutil.rmtree(base)
