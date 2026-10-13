#!/usr/bin/env python3
"""Check an existing vLLM OpenAI server and save JSON evidence (stdlib only).

Example:
    python scripts/smoke-api.py --url http://127.0.0.1:18030 \
        --model Qwen/Qwen3-0.6B --output artifacts/smoke-api.json

This script does not start a server or download model weights.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def positive_seconds(value):
    seconds = float(value)
    if not 0 < seconds < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return seconds


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request(url, timeout, payload=None):
    """Return the HTTP outcome, including failed requests, for the evidence file."""
    method = "GET" if payload is None else "POST"
    record = {"url": url, "method": method, "status": None}
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    started = time.monotonic()
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            record["status"] = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        record["status"] = error.code
        body = error.read().decode("utf-8", errors="replace")
        record["error"] = str(error)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        body = ""
        record["error"] = f"{type(error).__name__}: {error}"
    record["duration_seconds"] = round(time.monotonic() - started, 3)
    if body:
        try:
            record["response"] = json.loads(body)
        except json.JSONDecodeError:
            record["response_text"] = body[:4096]
    return record


def require_json_success(record, endpoint):
    if record["status"] != 200:
        raise RuntimeError(f"{endpoint} failed: {record.get('error', record['status'])}")
    response = record.get("response")
    if not isinstance(response, dict):
        raise RuntimeError(f"{endpoint} did not return a JSON object")
    return response


def check_server(args, report):
    deadline = time.monotonic() + args.health_timeout
    health = {"passed": False, "attempts": 0}
    report["checks"]["health"] = health
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"/health did not become ready within {args.health_timeout:g} seconds")
        health["attempts"] += 1
        health["last_response"] = request(
            args.url + "/health", min(args.request_timeout, remaining)
        )
        if health["last_response"]["status"] == 200:
            health["passed"] = True
            break
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(args.poll_interval, remaining))

    models = {"passed": False}
    report["checks"]["models"] = models
    models["http"] = request(args.url + "/v1/models", args.request_timeout)
    model_response = require_json_success(models["http"], "/v1/models")
    entries = model_response.get("data")
    if not isinstance(entries, list):
        raise RuntimeError("/v1/models response has no model data list")
    models["model_ids"] = [item.get("id") for item in entries if isinstance(item, dict)]
    if args.model not in models["model_ids"]:
        raise RuntimeError(f"Expected model {args.model!r}; served models: {models['model_ids']!r}")
    models["passed"] = True

    completion = {"passed": False}
    report["checks"]["chat_completion"] = completion
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly the word OK."}],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
        # Qwen3 otherwise may spend this small token budget on reasoning alone.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    completion["request"] = payload
    completion["http"] = request(
        args.url + "/v1/chat/completions", args.completion_timeout, payload
    )
    response = require_json_success(completion["http"], "/v1/chat/completions")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("Chat completion response has no choice")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Chat completion returned empty assistant content")
    completion["assistant_content"] = content
    completion["finish_reason"] = choices[0].get("finish_reason")
    completion["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18030", help="server root URL, without /v1")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="expected served model ID")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "smoke-api.json",
    )
    parser.add_argument("--health-timeout", type=positive_seconds, default=600.0)
    parser.add_argument("--request-timeout", type=positive_seconds, default=10.0)
    parser.add_argument("--completion-timeout", type=positive_seconds, default=180.0)
    parser.add_argument("--poll-interval", type=positive_seconds, default=2.0)
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    parsed_url = urllib.parse.urlsplit(args.url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        parser.error("--url must be an http:// or https:// server root URL")
    if parsed_url.query or parsed_url.fragment:
        parser.error("--url must not contain query parameters or a fragment")
    if not args.model.strip():
        parser.error("--model must not be empty")

    started = time.monotonic()
    report = {
        "started_at": utc_now(), "url": args.url, "model": args.model,
        "passed": False, "checks": {},
    }
    exit_code = 1
    try:
        check_server(args, report)
        report["passed"] = True
        exit_code = 0
    except KeyboardInterrupt:
        report["error"] = "Interrupted"
        exit_code = 130
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["finished_at"] = utc_now()
    report["duration_seconds"] = round(time.monotonic() - started, 3)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"Cannot save evidence to {args.output}: {error}", file=sys.stderr)
        return 1
    summary = {"passed": report["passed"], "output": str(args.output.resolve())}
    if "error" in report:
        summary["error"] = report["error"]
    print(json.dumps(summary, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
