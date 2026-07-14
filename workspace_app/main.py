from __future__ import annotations

import asyncio
import json
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .auth import AuthService
from .config import Settings
from .db import ACTIVE_RUN_STATUSES, Database, decode_json_columns, utc_now
from .filesystem import WorkspaceFilesystem, WorkspaceLimitError, WorkspacePathError
from .runner import RunManager
from .schemas import FeedbackUpsert, FileWrite, LoginRequest, RunCreate, WorkspaceCreate


def _workspace_public(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "userId": row["user_id"],
        "name": row["name"],
        "rootPath": "files/",
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _run_public(row: dict[str, Any], feedback: dict[str, Any] | None = None) -> dict[str, Any]:
    decoded = decode_json_columns(row) or {}
    return {
        "id": decoded["id"],
        "workspaceId": decoded["workspace_id"],
        "task": decoded["task"],
        "status": decoded["status"],
        "currentStep": decoded["current_step"],
        "finalResult": decoded.get("final_result"),
        "error": decoded.get("error"),
        "modelCalls": decoded.get("model_calls", 0),
        "toolCalls": decoded.get("tool_calls", 0),
        "durationMs": decoded.get("duration_ms"),
        "applyStatus": decoded.get("apply_status", "none"),
        "changedFiles": decoded.get("changed_files", []),
        "cancelRequested": decoded.get("cancel_requested", False),
        "createdAt": decoded["created_at"],
        "updatedAt": decoded["updated_at"],
        "startedAt": decoded.get("started_at"),
        "finishedAt": decoded.get("finished_at"),
        "feedback": None if not feedback else {
            "rating": feedback["rating"],
            "comment": feedback["comment"],
            "updatedAt": feedback["updated_at"],
        },
    }


def create_app(settings: Settings | None = None, run_manager: RunManager | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    filesystem = WorkspaceFilesystem(settings)
    auth = AuthService(settings)
    manager = run_manager or RunManager(database, filesystem, settings)
    workspace_locks: dict[str, threading.RLock] = {}
    locks_guard = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.database = database
        app.state.filesystem = filesystem
        app.state.auth = auth
        app.state.run_manager = manager
        app.state.settings = settings
        app.state.workspace_locks = workspace_locks
        manager.startup()
        try:
            yield
        finally:
            manager.shutdown()

    app = FastAPI(title="Workspace Agent MVP", version="0.1.0", lifespan=lifespan)
    app.state.database = database
    app.state.filesystem = filesystem
    app.state.auth = auth
    app.state.run_manager = manager
    app.state.settings = settings
    app.state.workspace_locks = workspace_locks

    def current_user(request: Request, workspace_session: str | None = Cookie(default=None)) -> dict[str, str]:
        user = request.app.state.auth.unsign(workspace_session)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
        return user

    def workspace_for_user(workspace_id: str, user_id: str) -> dict[str, Any]:
        row = database.fetchone("SELECT * FROM workspaces WHERE id=? AND user_id=?", (workspace_id, user_id))
        if not row:
            raise HTTPException(status_code=404, detail="Workspace not found.")
        return row

    def run_for_user(run_id: str, user_id: str) -> dict[str, Any]:
        row = database.fetchone(
            """SELECT r.* FROM agent_runs r JOIN workspaces w ON w.id=r.workspace_id
               WHERE r.id=? AND w.user_id=?""",
            (run_id, user_id),
        )
        if not row:
            raise HTTPException(status_code=404, detail="Run not found.")
        return row

    def lock_for(workspace_id: str) -> threading.RLock:
        with locks_guard:
            return workspace_locks.setdefault(workspace_id, threading.RLock())

    def reject_if_active(workspace_id: str) -> None:
        if database.active_run_for_workspace(workspace_id):
            raise HTTPException(status_code=409, detail="Workspace has an active or pending-review Run.")

    @app.get("/api/health")
    def health():
        return {"ok": True, "dockerRequired": True}

    @app.post("/api/auth/login")
    def login(payload: LoginRequest, response: Response):
        user = auth.authenticate(payload.username, payload.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        response.set_cookie(
            auth.cookie_name,
            auth.sign(user),
            max_age=settings.session_max_age,
            httponly=True,
            secure=settings.secure_cookies,
            samesite="strict",
            path="/",
        )
        return {"id": user.id, "username": user.username}

    @app.post("/api/auth/logout", status_code=204)
    def logout(response: Response):
        response.delete_cookie(auth.cookie_name, path="/")

    @app.get("/api/auth/me")
    def me(user=Depends(current_user)):
        return user

    @app.post("/api/workspaces", status_code=201)
    def create_workspace(payload: WorkspaceCreate, user=Depends(current_user)):
        workspace_id = str(uuid.uuid4())
        now = utc_now()
        root_path = str(filesystem.paths(workspace_id).root)
        row = {
            "id": workspace_id,
            "user_id": user["id"],
            "name": payload.name.strip(),
            "root_path": root_path,
            "created_at": now,
            "updated_at": now,
        }
        if not row["name"]:
            raise HTTPException(status_code=422, detail="Workspace name cannot be blank.")
        try:
            filesystem.create_workspace(workspace_id, _workspace_public(row))
            database.execute(
                "INSERT INTO workspaces(id,user_id,name,root_path,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (workspace_id, user["id"], row["name"], root_path, now, now),
            )
        except Exception:
            filesystem.delete_workspace(workspace_id)
            raise
        return _workspace_public(row)

    @app.get("/api/workspaces")
    def list_workspaces(user=Depends(current_user)):
        rows = database.fetchall("SELECT * FROM workspaces WHERE user_id=? ORDER BY updated_at DESC", (user["id"],))
        return [_workspace_public(row) for row in rows]

    @app.get("/api/workspaces/{workspace_id}")
    def get_workspace(workspace_id: str, user=Depends(current_user)):
        return _workspace_public(workspace_for_user(workspace_id, user["id"]))

    @app.delete("/api/workspaces/{workspace_id}", status_code=204)
    def delete_workspace(workspace_id: str, user=Depends(current_user)):
        workspace_for_user(workspace_id, user["id"])
        reject_if_active(workspace_id)
        with lock_for(workspace_id):
            database.execute("DELETE FROM workspaces WHERE id=?", (workspace_id,))
            filesystem.delete_workspace(workspace_id)

    @app.get("/api/workspaces/{workspace_id}/files")
    def list_workspace_files(workspace_id: str, path: str = Query(default="."), user=Depends(current_user)):
        workspace_for_user(workspace_id, user["id"])
        try:
            return filesystem.list_files(filesystem.paths(workspace_id).files, path)
        except (WorkspacePathError, WorkspaceLimitError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/workspaces/{workspace_id}/files/{file_path:path}")
    def get_workspace_file(workspace_id: str, file_path: str, user=Depends(current_user)):
        workspace_for_user(workspace_id, user["id"])
        root = filesystem.paths(workspace_id).files
        try:
            target = filesystem.resolve(root, file_path, allow_root=True)
            if target.is_dir():
                return filesystem.list_files(root, file_path)
            return {"path": file_path, "content": filesystem.read_file(root, file_path)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="File not found.") from exc
        except (WorkspacePathError, WorkspaceLimitError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/workspaces/{workspace_id}/files/{file_path:path}")
    def put_workspace_file(workspace_id: str, file_path: str, payload: FileWrite, user=Depends(current_user)):
        workspace = workspace_for_user(workspace_id, user["id"])
        reject_if_active(workspace_id)
        try:
            with lock_for(workspace_id):
                filesystem.write_file(filesystem.paths(workspace_id).files, file_path, payload.content)
                now = utc_now()
                database.execute("UPDATE workspaces SET updated_at=? WHERE id=?", (now, workspace_id))
                workspace["updated_at"] = now
                filesystem.update_metadata(workspace_id, _workspace_public(workspace))
            return {"path": file_path, "size": len(payload.content.encode("utf-8")), "updatedAt": now}
        except (WorkspacePathError, WorkspaceLimitError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/workspaces/{workspace_id}/runs", status_code=202)
    def create_run(workspace_id: str, payload: RunCreate, user=Depends(current_user)):
        workspace_for_user(workspace_id, user["id"])
        if payload.workspaceId != workspace_id:
            raise HTTPException(status_code=422, detail="workspaceId must match the URL.")
        task = payload.task.strip()
        if not task:
            raise HTTPException(status_code=422, detail="Task cannot be blank.")
        run_id = str(uuid.uuid4())
        now = utc_now()
        with database.transaction() as connection:
            placeholders = ",".join("?" for _ in ACTIVE_RUN_STATUSES)
            active = connection.execute(
                f"SELECT id FROM agent_runs WHERE workspace_id=? AND status IN ({placeholders}) LIMIT 1",
                (workspace_id, *ACTIVE_RUN_STATUSES),
            ).fetchone()
            if active:
                raise HTTPException(status_code=409, detail="Workspace already has an active or pending-review Run.")
            connection.execute(
                """INSERT INTO agent_runs(id,workspace_id,task,status,current_step,created_at,updated_at)
                   VALUES(?,?,?,'created',0,?,?)""",
                (run_id, workspace_id, task, now, now),
            )
        manager.start(run_id)
        return _run_public(database.fetchone("SELECT * FROM agent_runs WHERE id=?", (run_id,)))

    @app.get("/api/workspaces/{workspace_id}/runs")
    def list_runs(workspace_id: str, user=Depends(current_user)):
        workspace_for_user(workspace_id, user["id"])
        rows = database.fetchall("SELECT * FROM agent_runs WHERE workspace_id=? ORDER BY created_at DESC", (workspace_id,))
        return [_run_public(row) for row in rows]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, user=Depends(current_user)):
        row = run_for_user(run_id, user["id"])
        feedback = database.fetchone("SELECT * FROM feedback WHERE run_id=?", (run_id,))
        return _run_public(row, feedback)

    @app.get("/api/runs/{run_id}/diff")
    def get_run_diff(run_id: str, user=Depends(current_user)):
        run_for_user(run_id, user["id"])
        rows = database.fetchall(
            "SELECT path,change_type,diff,before_sha256,after_sha256 FROM file_changes WHERE run_id=? ORDER BY path",
            (run_id,),
        )
        return [{
            "path": row["path"],
            "changeType": row["change_type"],
            "diff": row["diff"],
            "beforeSha256": row["before_sha256"],
            "afterSha256": row["after_sha256"],
        } for row in rows]

    @app.get("/api/runs/{run_id}/events")
    async def stream_run_events(
        run_id: str,
        request: Request,
        user=Depends(current_user),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        after: int = Query(default=0, ge=0),
    ):
        run_for_user(run_id, user["id"])
        cursor = max(after, int(last_event_id or 0))

        async def generate():
            nonlocal cursor
            idle_terminal_polls = 0
            heartbeat_at = asyncio.get_running_loop().time()
            while not await request.is_disconnected():
                events = database.fetchall(
                    "SELECT * FROM run_events WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT 200",
                    (run_id, cursor),
                )
                for event in events:
                    cursor = event["sequence"]
                    payload = {
                        "sequence": event["sequence"],
                        "runId": run_id,
                        "type": event["type"],
                        "step": event["step"],
                        "payload": json.loads(event["payload_json"]),
                        "createdAt": event["created_at"],
                    }
                    data = json.dumps(payload, ensure_ascii=False).replace("\n", "\\n")
                    yield f"id: {cursor}\nevent: {event['type']}\ndata: {data}\n\n"
                row = database.fetchone("SELECT status FROM agent_runs WHERE id=?", (run_id,))
                if row and row["status"] in {"waiting_user", "completed", "failed", "cancelled"} and not events:
                    idle_terminal_polls += 1
                    if idle_terminal_polls >= 2:
                        break
                else:
                    idle_terminal_polls = 0
                now = asyncio.get_running_loop().time()
                if now - heartbeat_at >= 15:
                    yield ": heartbeat\n\n"
                    heartbeat_at = now
                await asyncio.sleep(0.35)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    def cancel_run(run_id: str, user=Depends(current_user)):
        row = run_for_user(run_id, user["id"])
        if row["status"] not in {"created", "running"}:
            raise HTTPException(status_code=409, detail="Run is not cancellable.")
        database.execute("UPDATE agent_runs SET cancel_requested=1, updated_at=? WHERE id=?", (utc_now(), run_id))
        manager.cancel(run_id)
        return {"id": run_id, "cancelRequested": True}

    def finish_review(run_id: str, user_id: str, action: str) -> dict[str, Any]:
        row = run_for_user(run_id, user_id)
        if row["status"] != "waiting_user" or row["apply_status"] != "pending":
            raise HTTPException(status_code=409, detail="Run is not waiting for file review.")
        with lock_for(row["workspace_id"]):
            row = run_for_user(run_id, user_id)
            if row["status"] != "waiting_user" or row["apply_status"] != "pending":
                raise HTTPException(status_code=409, detail="Run review was already completed.")
            if action == "applied":
                filesystem.apply_staging(row["workspace_id"], run_id)
            else:
                filesystem.discard_staging(row["workspace_id"], run_id)
            filesystem.remove_run_base(row["workspace_id"], run_id)
            if action == "applied":
                filesystem.discard_staging(row["workspace_id"], run_id)
            now = utc_now()
            database.execute(
                "UPDATE agent_runs SET status='completed', apply_status=?, updated_at=? WHERE id=?",
                (action, now, run_id),
            )
            database.execute("UPDATE workspaces SET updated_at=? WHERE id=?", (now, row["workspace_id"]))
            workspace = database.fetchone("SELECT * FROM workspaces WHERE id=?", (row["workspace_id"],))
            if workspace:
                filesystem.update_metadata(row["workspace_id"], _workspace_public(workspace))
            database.insert_event(run_id, "files_applied" if action == "applied" else "files_discarded", {"applyStatus": action})
        return _run_public(database.fetchone("SELECT * FROM agent_runs WHERE id=?", (run_id,)))

    @app.post("/api/runs/{run_id}/apply")
    def apply_run(run_id: str, user=Depends(current_user)):
        return finish_review(run_id, user["id"], "applied")

    @app.post("/api/runs/{run_id}/discard")
    def discard_run(run_id: str, user=Depends(current_user)):
        return finish_review(run_id, user["id"], "discarded")

    @app.put("/api/runs/{run_id}/feedback")
    def upsert_feedback(run_id: str, payload: FeedbackUpsert, user=Depends(current_user)):
        row = run_for_user(run_id, user["id"])
        if row["status"] != "completed":
            raise HTTPException(status_code=409, detail="Feedback is available after Run completion.")
        now = utc_now()
        database.execute(
            """INSERT INTO feedback(run_id,rating,comment,created_at,updated_at) VALUES(?,?,?,?,?)
               ON CONFLICT(run_id) DO UPDATE SET rating=excluded.rating,comment=excluded.comment,updated_at=excluded.updated_at""",
            (run_id, payload.rating, payload.comment, now, now),
        )
        return {"runId": run_id, "rating": payload.rating, "comment": payload.comment, "updatedAt": now}

    if settings.static_root.exists():
        assets = settings.static_root / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/{spa_path:path}", include_in_schema=False)
        def spa(spa_path: str):
            candidate = settings.static_root / spa_path
            if spa_path and candidate.is_file() and candidate.resolve().is_relative_to(settings.static_root.resolve()):
                return FileResponse(candidate)
            return FileResponse(settings.static_root / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def root():
            return JSONResponse({"name": "Workspace Agent MVP", "frontend": "Run npm build in frontend/."})

    return app
