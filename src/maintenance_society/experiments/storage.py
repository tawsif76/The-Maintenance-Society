"""Atomic files, an operating-system run lock, and an at-most-once request journal."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from maintenance_society.agents.ollama_client import ChatResult, OllamaError, ContextCapacityError, generation_aborted


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value, immutable=False):
    """Flush a complete temporary file before atomically publishing it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable record differs: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RunLock:
    """OS locks are released even if the process is killed. Do not delete the lock file."""

    def __init__(self, directory):
        self.path = Path(directory) / "run.lock"
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            self.handle.seek(0, os.SEEK_END)
            if self.handle.tell() == 0:
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Another process is using this run directory") from error
        return self

    def __exit__(self, *unused):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class UncertainRequest(OllamaError):
    pass


def check_context(result):
    """Require reported input size and room for the entire requested output budget."""
    if generation_aborted(result.response):
        # There is no accepted decision and no terminal token-usage report. The controller
        # must reject this candidate. A successful correction still requires full checks.
        return
    tokens = result.response.get("prompt_eval_count")
    options = result.request["options"]
    if type(tokens) is not int or tokens < 1:
        raise ContextCapacityError("Runtime did not report an input token count")
    if tokens + options["num_predict"] > options["num_ctx"]:
        raise ContextCapacityError("Complete input plus output budget exceeds context capacity")


class JournalClient:
    """Cache each exact request inside one episode before it can update the workshop.

    A request marker without a response is ambiguous. It is never automatically resent.
    The enclosing run must hold RunLock throughout all use of this journal.
    """

    def __init__(self, client, directory):
        self.client = client
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def chat(self, messages, schema, phase, seed):
        payload = self.client.build_payload(messages, schema, phase, seed)
        key = digest(payload)
        request_path = self.directory / f"{key}.request.json"
        response_path = self.directory / f"{key}.response.json"
        if response_path.exists():
            saved = read_json(response_path)
            if not request_path.exists() or read_json(request_path)["payload"] != payload:
                raise ValueError("Journal request/response mismatch")
            if saved["request_hash"] != key:
                raise ValueError("Journal response hash mismatch")
            result = ChatResult(saved["content"], payload, saved["response"], saved["elapsed_seconds"])
        else:
            if request_path.exists():
                raise UncertainRequest(
                    "A request was started but no complete response was saved. "
                    "Its outcome is unknown; this episode will not resend it.")
            atomic_json(request_path, {"request_hash": key, "phase": phase,
                                      "started_at_utc": utc_now(), "payload": payload}, immutable=True)
            try:
                result = self.client.chat(messages, schema, phase, seed)
            except BaseException as error:
                atomic_json(self.directory / f"{key}.error.json", {
                    "request_hash": key, "recorded_at_utc": utc_now(),
                    "exception_type": type(error).__name__, "message": str(error),
                }, immutable=True)
                raise
            if result.request != payload:
                raise ValueError("Inference client changed the journaled request")
            atomic_json(response_path, {
                "request_hash": key, "completed_at_utc": utc_now(),
                "content": result.content, "response": result.response,
                "elapsed_seconds": result.elapsed_seconds,
            }, immutable=True)
        # Even invalid content is saved before validating its context or JSON.
        check_context(result)
        return result

    def record_attempt(self, attempt):
        record = asdict(attempt)
        record.pop("result")
        record["request_hash"] = digest(attempt.result.request)
        atomic_json(self.directory / f"{record['request_hash']}.validation.json", record, immutable=True)

    def prepare_correction(self, result):
        if not generation_aborted(result.response):
            return None
        key = digest(result.request)
        event_id = uuid.uuid4().hex
        prefix = self.directory / f"{key}.{event_id}.reset"
        atomic_json(str(prefix) + ".start.json", {"request_hash": key, "started_at_utc": utc_now()}, immutable=True)
        response = self.client.prepare_correction(result)
        atomic_json(str(prefix) + ".done.json", {"request_hash": key, "completed_at_utc": utc_now(),
                                               "reset": response}, immutable=True)
        return response

    def statistics(self):
        responses = [read_json(path) for path in self.directory.glob("*.response.json")]
        validations = [read_json(path) for path in self.directory.glob("*.validation.json")]
        requests = [read_json(path) for path in self.directory.glob("*.request.json")]
        requested = len(requests)
        invalid = sum(not item["valid"] for item in validations)
        counters = {}
        for field in ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration"):
            values = [item["response"].get(field) for item in responses]
            counters[field] = sum(values) if all(type(v) is int for v in values) else None
        return {
            "requests_started": requested, "responses_saved": len(responses),
            "requests_without_saved_response": requested - len(responses),
            "validated_responses": len(validations), "invalid_responses": invalid,
            "invalid_response_rate": invalid / len(validations) if validations else None,
            # A correction can fail before validation. Count its durable start as well.
            "correction_requests": sum(len(item["payload"]["messages"]) > 2 for item in requests),
            "generation_repetition_responses": sum(generation_aborted(item["response"]) for item in responses),
            "model_resets_before_correction": len(list(self.directory.glob("*.reset.done.json"))),
            "responses_without_token_usage": sum(
                type(item["response"].get("prompt_eval_count")) is not int
                or type(item["response"].get("eval_count")) is not int for item in responses),
            "known_prompt_eval_count": sum(item["response"]["prompt_eval_count"] for item in responses
                                            if type(item["response"].get("prompt_eval_count")) is int),
            "known_eval_count": sum(item["response"]["eval_count"] for item in responses
                                      if type(item["response"].get("eval_count")) is int),
            "elapsed_inference_seconds": sum(item["elapsed_seconds"] for item in responses),
            **counters,
        }
