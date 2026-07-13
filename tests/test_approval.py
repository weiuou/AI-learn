from agent import approval


def test_non_interactive_override_disables_tty_approval(monkeypatch):
    monkeypatch.setenv("AGENT_NON_INTERACTIVE", "1")
    monkeypatch.setattr(approval.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(approval.sys.stdout, "isatty", lambda: True)
    assert approval.is_interactive() is False
    assert approval.request_cli_approval("run_shell", {}, "high", "test") is False
