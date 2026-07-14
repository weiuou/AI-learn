import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workspace_app.auth import hash_password
from workspace_app.config import Settings
from workspace_app.filesystem import WorkspaceFilesystem, WorkspacePathError
from workspace_app.main import create_app


class FakeRunManager:
    def __init__(self):
        self.started = []
        self.cancelled = []

    def startup(self):
        pass

    def shutdown(self):
        pass

    def start(self, run_id):
        self.started.append(run_id)

    def cancel(self, run_id):
        self.cancelled.append(run_id)
        return True


def make_settings(tmp_path: Path) -> Settings:
    users = [
        {"id": "user-a", "username": "alice", "passwordHash": hash_password("alice-pass", salt=b"alice-salt-12345")},
        {"id": "user-b", "username": "bob", "passwordHash": hash_password("bob-pass", salt=b"bob-salt-123456")},
    ]
    return Settings(
        data_root=tmp_path / "data",
        host_data_root=tmp_path / "data",
        static_root=tmp_path / "missing-static",
        environment="test",
        session_secret="test-secret-that-is-long-enough-for-tests",
        session_max_age=3600,
        secure_cookies=False,
        invite_users_json=json.dumps(users),
        sandbox_image="workspace-sandbox:test",
        sandbox_uid=1000,
        sandbox_gid=1000,
        max_file_bytes=1024 * 1024,
        max_workspace_bytes=50 * 1024 * 1024,
        max_tool_output_bytes=64 * 1024,
        max_parallel_runs=2,
        shell_timeout_seconds=60,
        run_timeout_seconds=900,
    )


@pytest.fixture()
def app_client(tmp_path):
    manager = FakeRunManager()
    app = create_app(make_settings(tmp_path), run_manager=manager)
    with TestClient(app) as client:
        yield app, client, manager


def login(client, username="alice", password="alice-pass"):
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200


def create_workspace(client, name):
    response = client.post("/api/workspaces", json={"name": name})
    assert response.status_code == 201
    return response.json()


def test_workspace_files_are_isolated_and_persist(app_client):
    _, client, _ = app_client
    assert client.get("/api/workspaces").status_code == 401
    login(client)
    first = create_workspace(client, "First")
    second = create_workspace(client, "Second")

    assert client.put(f"/api/workspaces/{first['id']}/files/note.txt", json={"content": "alpha"}).status_code == 200
    assert client.put(f"/api/workspaces/{second['id']}/files/note.txt", json={"content": "beta"}).status_code == 200
    assert client.get(f"/api/workspaces/{first['id']}/files/note.txt").json()["content"] == "alpha"
    assert client.get(f"/api/workspaces/{second['id']}/files/note.txt").json()["content"] == "beta"

    listed = client.get("/api/workspaces").json()
    assert {item["id"] for item in listed} == {first["id"], second["id"]}
    assert all(item["rootPath"] == "files/" for item in listed)


def test_cross_user_resources_return_not_found(app_client):
    _, client, _ = app_client
    login(client)
    workspace = create_workspace(client, "Alice only")
    client.post("/api/auth/logout")
    login(client, "bob", "bob-pass")
    assert client.get(f"/api/workspaces/{workspace['id']}").status_code == 404
    assert client.get("/api/workspaces").json() == []


def test_run_review_applies_staging_and_feedback(app_client):
    app, client, manager = app_client
    login(client)
    workspace = create_workspace(client, "Review")
    client.put(f"/api/workspaces/{workspace['id']}/files/app.txt", json={"content": "before\n"})

    created = client.post(
        f"/api/workspaces/{workspace['id']}/runs",
        json={"workspaceId": workspace["id"], "task": "Update app.txt"},
    )
    assert created.status_code == 202
    run = created.json()
    assert manager.started == [run["id"]]
    assert client.put(f"/api/workspaces/{workspace['id']}/files/blocked.txt", json={"content": "no"}).status_code == 409

    filesystem = app.state.filesystem
    _, staging = filesystem.prepare_run(workspace["id"], run["id"])
    filesystem.write_file(staging, "app.txt", "after\n")
    app.state.database.execute(
        "UPDATE agent_runs SET status='waiting_user', apply_status='pending', changed_files_json='[\"app.txt\"]' WHERE id=?",
        (run["id"],),
    )

    assert client.get(f"/api/workspaces/{workspace['id']}/files/app.txt").json()["content"] == "before\n"
    applied = client.post(f"/api/runs/{run['id']}/apply")
    assert applied.status_code == 200
    assert applied.json()["applyStatus"] == "applied"
    assert client.get(f"/api/workspaces/{workspace['id']}/files/app.txt").json()["content"] == "after\n"

    feedback = client.put(f"/api/runs/{run['id']}/feedback", json={"rating": "up", "comment": "worked"})
    assert feedback.status_code == 200
    restored = client.get(f"/api/runs/{run['id']}").json()
    assert restored["feedback"]["comment"] == "worked"


def test_run_review_discard_keeps_canonical(app_client):
    app, client, _ = app_client
    login(client)
    workspace = create_workspace(client, "Discard")
    client.put(f"/api/workspaces/{workspace['id']}/files/app.txt", json={"content": "keep\n"})
    run = client.post(
        f"/api/workspaces/{workspace['id']}/runs",
        json={"workspaceId": workspace["id"], "task": "Replace app.txt"},
    ).json()
    filesystem = app.state.filesystem
    _, staging = filesystem.prepare_run(workspace["id"], run["id"])
    filesystem.write_file(staging, "app.txt", "discard me\n")
    app.state.database.execute(
        "UPDATE agent_runs SET status='waiting_user', apply_status='pending' WHERE id=?",
        (run["id"],),
    )
    discarded = client.post(f"/api/runs/{run['id']}/discard")
    assert discarded.status_code == 200
    assert discarded.json()["applyStatus"] == "discarded"
    assert client.get(f"/api/workspaces/{workspace['id']}/files/app.txt").json()["content"] == "keep\n"


def test_restart_recovery_marks_only_active_runs_failed(app_client):
    app, client, _ = app_client
    login(client)
    workspace = create_workspace(client, "Recovery")
    run = client.post(
        f"/api/workspaces/{workspace['id']}/runs",
        json={"workspaceId": workspace["id"], "task": "Interrupted"},
    ).json()
    recovered = app.state.database.recover_interrupted_runs()
    assert recovered == [run["id"]]
    assert client.get(f"/api/runs/{run['id']}").json()["status"] == "failed"


def test_sse_replays_persisted_events(app_client):
    app, client, _ = app_client
    login(client)
    workspace = create_workspace(client, "Events")
    created = client.post(
        f"/api/workspaces/{workspace['id']}/runs",
        json={"workspaceId": workspace["id"], "task": "Say hello"},
    ).json()
    app.state.database.insert_event(created["id"], "agent_message", {"content": "hello"}, 1)
    app.state.database.execute("UPDATE agent_runs SET status='completed' WHERE id=?", (created["id"],))
    response = client.get(f"/api/runs/{created['id']}/events")
    assert response.status_code == 200
    assert "event: agent_message" in response.text
    assert '"content": "hello"' in response.text


def test_path_traversal_and_symlinks_are_rejected(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_directories()
    filesystem = WorkspaceFilesystem(settings)
    workspace_id = "11111111-1111-1111-1111-111111111111"
    paths = filesystem.create_workspace(workspace_id, {"id": workspace_id})
    with pytest.raises(WorkspacePathError):
        filesystem.resolve(paths.files, "../outside.txt", allow_missing=True)
    with pytest.raises(WorkspacePathError):
        filesystem.resolve(paths.files, "/etc/passwd")
    target = tmp_path / "outside.txt"
    target.write_text("secret", encoding="utf-8")
    (paths.files / "escape.txt").symlink_to(target)
    with pytest.raises(WorkspacePathError):
        filesystem.read_file(paths.files, "escape.txt")
    with pytest.raises(WorkspacePathError):
        filesystem.search_files(paths.files, "secret", glob="../*")
