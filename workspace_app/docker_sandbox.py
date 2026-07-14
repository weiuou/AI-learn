from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from .config import Settings

try:
    import docker
except ImportError:  # pragma: no cover - exercised through the unavailable branch
    docker = None


class SandboxUnavailable(RuntimeError):
    pass


class DockerSandbox:
    def __init__(
        self,
        settings: Settings,
        run_id: str,
        host_staging_path: Path,
        output_callback: Callable[[str, str], None] | None = None,
        client=None,
    ):
        self.settings = settings
        self.run_id = run_id
        self.host_staging_path = Path(host_staging_path)
        self.output_callback = output_callback
        self.client = client
        self.container = None
        self._destroyed = threading.Event()

    def start(self) -> None:
        if docker is None and self.client is None:
            raise SandboxUnavailable("The Docker Python package is not installed.")
        if not self.host_staging_path.exists():
            raise SandboxUnavailable(f"Host staging path does not exist: {self.host_staging_path}")
        try:
            self.client = self.client or docker.from_env()
            self.client.ping()
            self.container = self.client.containers.run(
                self.settings.sandbox_image,
                command=["sleep", "infinity"],
                detach=True,
                name=f"workspace-run-{self.run_id}",
                hostname=f"run-{self.run_id[:12]}",
                network_disabled=True,
                read_only=True,
                user=f"{self.settings.sandbox_uid}:{self.settings.sandbox_gid}",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                mem_limit="512m",
                memswap_limit="512m",
                nano_cpus=1_000_000_000,
                pids_limit=128,
                working_dir="/workspace",
                volumes={str(self.host_staging_path): {"bind": "/workspace", "mode": "rw"}},
                tmpfs={"/tmp": "rw,noexec,nosuid,nodev,size=64m"},
                environment={"HOME": "/workspace", "LANG": "C.UTF-8"},
                labels={"workspace-mvp.managed": "true", "workspace-mvp.run-id": self.run_id},
            )
        except Exception as exc:
            raise SandboxUnavailable(
                "Docker Engine is unavailable or the sandbox image could not be started. "
                f"Build {self.settings.sandbox_image} and verify docker info. Details: {exc}"
            ) from exc

    def exec(self, command: str) -> dict:
        if self.container is None or self._destroyed.is_set():
            raise SandboxUnavailable("Sandbox is not running.")
        timeout = str(self.settings.shell_timeout_seconds)
        try:
            created = self.client.api.exec_create(
                self.container.id,
                cmd=["/usr/bin/timeout", "--signal=KILL", f"{timeout}s", "/bin/bash", "-lc", command],
                workdir="/workspace",
                user=f"{self.settings.sandbox_uid}:{self.settings.sandbox_gid}",
                environment={"HOME": "/workspace", "LANG": "C.UTF-8"},
            )
            exec_id = created["Id"]
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            captured = 0
            truncated = False
            stream = self.client.api.exec_start(exec_id, stream=True, demux=True)
            for item in stream:
                if self._destroyed.is_set():
                    break
                stdout_bytes, stderr_bytes = item if isinstance(item, tuple) else (item, None)
                for stream_name, raw, target in (
                    ("stdout", stdout_bytes, stdout_parts),
                    ("stderr", stderr_bytes, stderr_parts),
                ):
                    if not raw:
                        continue
                    text = raw.decode("utf-8", errors="replace")
                    remaining = self.settings.max_tool_output_bytes - captured
                    accepted = text[: max(remaining, 0)]
                    if accepted:
                        target.append(accepted)
                        captured += len(accepted.encode("utf-8"))
                        if self.output_callback:
                            self.output_callback(stream_name, accepted)
                    if accepted != text:
                        truncated = True
            info = self.client.api.exec_inspect(exec_id)
            exit_code = int(info.get("ExitCode") if info.get("ExitCode") is not None else -1)
            return {
                "returncode": exit_code,
                "stdout": "".join(stdout_parts),
                "stderr": "".join(stderr_parts),
                "truncated": truncated,
                "timedOut": exit_code == 124,
            }
        except Exception as exc:
            if self._destroyed.is_set():
                raise SandboxUnavailable("Sandbox was cancelled.") from exc
            raise SandboxUnavailable(f"Sandbox command failed: {exc}") from exc

    def destroy(self) -> None:
        self._destroyed.set()
        container = self.container
        self.container = None
        if container is None:
            return
        try:
            container.remove(force=True)
        except Exception:
            pass


class DockerSandboxFactory:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.client = client

    def create(self, run_id: str, host_staging_path: Path, output_callback=None) -> DockerSandbox:
        return DockerSandbox(
            self.settings,
            run_id,
            host_staging_path,
            output_callback=output_callback,
            client=self.client,
        )

    def cleanup_orphans(self) -> None:
        if docker is None and self.client is None:
            return
        try:
            client = self.client or docker.from_env()
            for container in client.containers.list(all=True, filters={"label": "workspace-mvp.managed=true"}):
                container.remove(force=True)
        except Exception:
            return
