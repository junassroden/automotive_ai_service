"""The Back-End's chat flow on a fixture database, for tests and the evaluation.

    ask(message, role, user_id) runs what ChatController does for rental mode: the AI
    service plans, Back-End/tests/chat_retrieval_cli.php runs the plan with the real
    ChatRetrieval code on Back-End/tests/fixtures/chat_fixture_db.json, and the AI service
    answers from the results. Requires the PHP CLI (PATH or PHP_BINARY) and Django set up.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

from ai_api.assistant import answer_chat


BACKEND = Path(__file__).resolve().parents[2] / "Back-End"
RETRIEVAL_CLI = BACKEND / "tests" / "chat_retrieval_cli.php"
FIXTURE_DB = BACKEND / "tests" / "fixtures" / "chat_fixture_db.json"
# Fixture users: 1 is the administrator, 2 and 3 are customers.
DEFAULT_USERS = {"admin": 1, "customer": 2}


def php_binary():
    php = os.environ.get("PHP_BINARY") or shutil.which("php")
    if not php:
        raise SystemExit("PHP CLI not found (set PHP_BINARY); the Back-End retrieval code cannot run without it.")
    return php


def backend(requests, fixture_path=FIXTURE_DB):
    completed = subprocess.run([php_binary(), str(RETRIEVAL_CLI)], capture_output=True, text=True, encoding="utf-8",
                               input=json.dumps({"fixture_path": str(fixture_path), "requests": requests}), check=True)
    return json.loads(completed.stdout)


def ask(message, role="customer", user_id=None, conversation_id=None, fixture_overrides=None):
    """Returns (final response, plan or None, retrieval results or None)."""
    user_id = DEFAULT_USERS[role] if user_id is None else user_id
    vocabulary = backend([{"op": "vocabulary", "is_admin": role == "admin", "fixture_overrides": fixture_overrides or {}}])[0]
    request = dict(viewer_role=role, conversation_id=conversation_id, conversation_scope=f"{role}:{user_id}",
                   retrieval_protocol=1, vocabulary=vocabulary)
    first = answer_chat(message, **request)
    if first.get("status") != "needs_data":
        return first, None, None
    results = backend([{"op": "execute", "viewer": {"role": role, "user_id": user_id}, "plan": first["plan"],
                        "fixture_overrides": fixture_overrides or {}}])[0]
    final = answer_chat(message, retrieval={"plan": first["plan"], "results": results, "data_as_of": "fixture"}, **request)
    return final, first["plan"], results
