import os
import sys


def is_interactive():
    if os.getenv("AGENT_NON_INTERACTIVE", "").strip().casefold() in {"1", "true", "yes"}:
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def request_cli_approval(tool_name, args, risk_level, risk_reason):
    if not is_interactive():
        return False

    print()
    print(f"Agent wants to run tool: {tool_name}")
    print(f"Risk: {risk_level}")
    print(f"Reason: {risk_reason}")
    print(f"Args: {args}")
    answer = input("Approve? [y/N] ").strip().casefold()
    return answer in {"y", "yes"}
