from pathlib import Path

from workspace_app.docker_sandbox import DockerSandbox

from test_workspace_api import make_settings


class FakeContainer:
    id = "container-id"

    def __init__(self):
        self.removed = False

    def remove(self, force=False):
        self.removed = force


class FakeContainers:
    def __init__(self):
        self.kwargs = None
        self.container = FakeContainer()

    def run(self, image, **kwargs):
        self.image = image
        self.kwargs = kwargs
        return self.container


class FakeClient:
    def __init__(self):
        self.containers = FakeContainers()

    def ping(self):
        return True


def test_docker_sandbox_enforces_security_options(tmp_path: Path):
    settings = make_settings(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    client = FakeClient()
    sandbox = DockerSandbox(settings, "run-123", staging, client=client)
    sandbox.start()

    options = client.containers.kwargs
    assert options["network_disabled"] is True
    assert options["read_only"] is True
    assert options["cap_drop"] == ["ALL"]
    assert options["security_opt"] == ["no-new-privileges:true"]
    assert options["mem_limit"] == "512m"
    assert options["memswap_limit"] == "512m"
    assert options["nano_cpus"] == 1_000_000_000
    assert options["pids_limit"] == 128
    assert options["volumes"][str(staging)]["bind"] == "/workspace"
    assert options["volumes"][str(staging)]["mode"] == "rw"
    assert "DOCKER_HOST" not in options["environment"]

    sandbox.destroy()
    assert client.containers.container.removed is True
