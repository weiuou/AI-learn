from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_root: Path
    host_data_root: Path
    static_root: Path
    environment: str
    session_secret: str
    session_max_age: int
    secure_cookies: bool
    invite_users_json: str
    sandbox_image: str
    sandbox_uid: int
    sandbox_gid: int
    max_file_bytes: int
    max_workspace_bytes: int
    max_tool_output_bytes: int
    max_parallel_runs: int
    shell_timeout_seconds: int
    run_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        data_root = Path(os.getenv("APP_DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser().resolve()
        host_root = Path(os.getenv("HOST_DATA_ROOT", str(data_root))).expanduser()
        environment = os.getenv("APP_ENV", "development").strip().casefold()
        return cls(
            data_root=data_root,
            host_data_root=host_root,
            static_root=Path(os.getenv("APP_STATIC_ROOT", str(PROJECT_ROOT / "frontend" / "dist"))).expanduser().resolve(),
            environment=environment,
            session_secret=os.getenv("SESSION_SECRET", "development-only-change-me"),
            session_max_age=int(os.getenv("SESSION_MAX_AGE", "604800")),
            secure_cookies=_env_bool("SECURE_COOKIES", environment == "production"),
            invite_users_json=os.getenv("INVITE_USERS_JSON", ""),
            sandbox_image=os.getenv("SANDBOX_IMAGE", "workspace-sandbox:latest"),
            sandbox_uid=int(os.getenv("SANDBOX_UID", "1000")),
            sandbox_gid=int(os.getenv("SANDBOX_GID", "1000")),
            max_file_bytes=int(os.getenv("MAX_FILE_BYTES", str(1024 * 1024))),
            max_workspace_bytes=int(os.getenv("MAX_WORKSPACE_BYTES", str(50 * 1024 * 1024))),
            max_tool_output_bytes=int(os.getenv("MAX_TOOL_OUTPUT_BYTES", str(64 * 1024))),
            max_parallel_runs=int(os.getenv("MAX_PARALLEL_RUNS", "2")),
            shell_timeout_seconds=int(os.getenv("SHELL_TIMEOUT_SECONDS", "60")),
            run_timeout_seconds=int(os.getenv("RUN_TIMEOUT_SECONDS", "900")),
        )

    def validate(self) -> None:
        if self.environment == "production":
            if self.session_secret == "development-only-change-me" or len(self.session_secret) < 32:
                raise RuntimeError("SESSION_SECRET must contain at least 32 characters in production.")
            if not self.invite_users_json:
                raise RuntimeError("INVITE_USERS_JSON is required in production.")
        if self.invite_users_json:
            parsed = json.loads(self.invite_users_json)
            if not isinstance(parsed, list) or not parsed:
                raise RuntimeError("INVITE_USERS_JSON must be a non-empty JSON array.")

    @property
    def database_path(self) -> Path:
        return self.data_root / "workspace.sqlite3"

    @property
    def workspaces_root(self) -> Path:
        return self.data_root / "workspaces"

    def ensure_directories(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.workspaces_root.mkdir(parents=True, exist_ok=True)
