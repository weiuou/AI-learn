import json
import os
import re
from collections import Counter

from evaluators import evaluate_task
from failure_classifier import classify_failure


def _safe_task_id(task_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_") or "task"


def _shorten(value, limit=160):
    text = value or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _exception_error_type(error):
    error_text = f"{error.__class__.__name__}: {error}"
    if "timeout" in error_text.casefold() or "timed out" in error_text.casefold():
        return "REQUEST_TIMEOUT"
    return "RUNTIME_ERROR"


def _trace_has_error_type(trace, error_type):
    for event in trace.get("events", []):
        attrs = event.get("attributes") or event.get("data") or {}
        observation = attrs.get("observation") or attrs.get("result")
        error = attrs.get("error")

        if isinstance(error, dict) and error.get("error_type") == error_type:
            return True
        if isinstance(observation, dict) and observation.get("error_type") == error_type:
            return True
    return False


def load_tasks(tasks_path):
    tasks = []

    with open(tasks_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                task = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{tasks_path}:{line_number} is not valid JSON: {e}") from e

            if not isinstance(task, dict):
                raise ValueError(f"{tasks_path}:{line_number} must be a JSON object.")
            if not task.get("id"):
                raise ValueError(f"{tasks_path}:{line_number} is missing required field: id.")
            if not task.get("prompt"):
                raise ValueError(f"{tasks_path}:{line_number} is missing required field: prompt.")

            tasks.append(task)

    return tasks


def _new_trace(task):
    from agent import add_event, estimate_text_tokens, now

    trace = {
        "schema_version": "agent-harness-trace-v1",
        "task_id": task["id"],
        "task": task["prompt"],
        "user_goal": task["prompt"],
        "started_at": now(),
        "finished_at": None,
        "events": [],
    }
    add_event(
        trace,
        "task_started",
        {
            "task_id": task["id"],
            "user_goal": task["prompt"],
            "token_estimate": estimate_text_tokens(task["prompt"]),
        },
    )
    return trace


def run_eval(tasks_path, out_path):
    from agent import add_event, estimate_text_tokens, now, run_agent, save_trace, summarize_usage

    tasks = load_tasks(tasks_path)
    report_dir = os.path.dirname(out_path) or "."
    trace_dir = os.path.join(report_dir, "evals")
    os.makedirs(trace_dir, exist_ok=True)

    results = []

    for task in tasks:
        task_id = task["id"]
        max_steps = task.get("max_steps", 50)
        trace_path = os.path.join(trace_dir, f"{_safe_task_id(task_id)}.json")
        trace = _new_trace(task)
        final_answer = ""

        try:
            final_answer = run_agent(task["prompt"], trace, max_steps=max_steps)
        except Exception as e:
            final_answer = f"任务运行失败: {e}"
            error_type = _exception_error_type(e)
            add_event(
                trace,
                "error",
                {
                    "task_id": task_id,
                    "error_type": error_type,
                    "message": str(e),
                    "token_estimate": estimate_text_tokens(str(e)),
                },
            )
            add_event(
                trace,
                "final_answer",
                {
                    "task_id": task_id,
                    "user_goal": task["prompt"],
                    "answer": final_answer,
                    "exit_reason": "runtime_error",
                    "token_estimate": estimate_text_tokens(final_answer),
                },
            )
        finally:
            trace["finished_at"] = now()
            trace["usage_summary"] = summarize_usage(trace)
            save_trace(trace, trace_path)

        checks = evaluate_task(trace, final_answer, task)
        passed = all(checks.values()) if checks else True
        failure_reason = None if passed else classify_failure(trace, checks, task)

        result = {
            "task_id": task_id,
            "category": task.get("category"),
            "passed": passed,
            "checks": checks,
            "failure_reason": failure_reason,
            "trace_file": trace_path,
            "final_answer_preview": _shorten(final_answer),
        }
        results.append(result)

        status = "PASS" if passed else f"FAIL {failure_reason}"
        print(f"[{status}] {task_id} -> {trace_path}", flush=True)

    passed_count = sum(1 for result in results if result["passed"])
    failed_count = len(results) - passed_count
    failure_reasons = Counter(
        result["failure_reason"]
        for result in results
        if not result["passed"] and result.get("failure_reason")
    )
    security_results = [
        result
        for result in results
        if result.get("category") == "security"
    ]
    security_task_ids = {result["task_id"] for result in security_results}
    permission_denied_hits = 0
    for task in tasks:
        if task.get("id") not in security_task_ids:
            continue
        trace_file = os.path.join(trace_dir, f"{_safe_task_id(task['id'])}.json")
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                task_trace = json.load(f)
        except FileNotFoundError:
            continue
        if _trace_has_error_type(task_trace, "PERMISSION_DENIED"):
            permission_denied_hits += 1

    report = {
        "schema_version": "agent-harness-eval-report-v1",
        "task_file": tasks_path,
        "generated_at": now(),
        "total": len(results),
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": passed_count / len(results) if results else 0,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "security_summary": {
            "total": len(security_results),
            "passed": sum(1 for result in security_results if result["passed"]),
            "permission_denied_hits": permission_denied_hits,
        },
        "results": results,
    }

    os.makedirs(report_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Eval report saved to {out_path}", flush=True)
    return report


def parse_eval_args(argv):
    if not argv:
        raise ValueError("Usage: python3 agent.py eval <tasks.jsonl> --out <report.json>")

    args = list(argv)
    tasks_path = args.pop(0)
    out_path = "runs/eval_report.json"

    if "--out" in args:
        index = args.index("--out")
        if index + 1 >= len(args):
            raise ValueError("--out requires a path.")
        out_path = args[index + 1]
        del args[index : index + 2]

    if args:
        raise ValueError(f"Unknown eval arguments: {' '.join(args)}")

    return tasks_path, out_path
