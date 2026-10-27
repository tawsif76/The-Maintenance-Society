"""Native local Ollama API. Uses a single model and serialized requests."""

from dataclasses import dataclass, fields
import json
import math
from threading import Lock
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

REPETITION_ERROR = "prediction aborted, token repeat limit reached"


def generation_aborted(response):
    """Only this explicit provider termination is eligible for output correction."""
    return (isinstance(response, dict) and response.get("transport_outcome") == "generation_aborted"
            and response.get("error") == REPETITION_ERROR and response.get("done") is False
            and response.get("done_reason") == "repeat_limit")


def aborted_response(content, events, http_status=None):
    value = {"done": False, "done_reason": "repeat_limit", "error": REPETITION_ERROR,
             "transport_outcome": "generation_aborted",
             "message": {"role": "assistant", "content": content}, "stream_events": events}
    if http_status is not None:
        value["http_status"] = http_status
    return value


def read_chat_stream(lines):
    """Keep raw events and partial text; never invent a successful terminal event."""
    events, contents, thinking = [], [], []
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict):
            raise OllamaError("Unexpected non-object streaming event")
        events.append(event)
        if "error" in event:
            if event["error"] == REPETITION_ERROR:
                if any(thinking):
                    raise OllamaError("Unexpected thinking output; check model mode")
                return aborted_response("".join(contents), events)
            raise OllamaError(f"Ollama streaming error: {event['error']}")
        message = event.get("message", {})
        if not isinstance(message, dict):
            raise OllamaError("Invalid streaming message")
        for field, fragments in (("content", contents), ("thinking", thinking)):
            fragment = message.get(field, "")
            if not isinstance(fragment, str):
                raise OllamaError("Invalid streaming text fragment")
            fragments.append(fragment)
        if event.get("done") is True:
            return {**event, "message": {"role": "assistant", "content": "".join(contents),
                                        "thinking": "".join(thinking)}, "stream_events": events}
    raise OllamaError("Ollama stream ended without a terminal event; outcome is uncertain")


class OllamaError(RuntimeError):
    pass


class ContextCapacityError(OllamaError):
    pass


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise OllamaError("Redirects are disabled for the local inference client")


@dataclass(frozen=True)
class GenerationSettings:
    context_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.0
    max_action_output_tokens: int = 64
    max_proposal_output_tokens: int = 256
    max_recognition_output_tokens: int = 256

    def __post_init__(self):
        for name in ("context_tokens", "top_k", "max_action_output_tokens",
                     "max_proposal_output_tokens", "max_recognition_output_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("temperature", "top_p", "repeat_penalty"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between zero and two")
        if not 0 < self.top_p <= 1 or self.repeat_penalty <= 0:
            raise ValueError("Invalid sampling settings")
        if max(self.max_action_output_tokens, self.max_proposal_output_tokens,
               self.max_recognition_output_tokens) >= self.context_tokens:
            raise ValueError("Context must leave room for the output budget")

    @classmethod
    def from_config(cls, generation):
        names = {field.name for field in fields(cls)}
        if set(generation) - names - {"status"}:
            raise ValueError("Unknown generation configuration fields")
        if any(name not in generation or generation[name] is None for name in names):
            raise ValueError("All generation settings must be explicit")
        return cls(**{name: generation[name] for name in names})

    def options(self, phase, seed):
        limit = {"decision": self.max_action_output_tokens,
                 "proposal": self.max_proposal_output_tokens,
                 "recognition": self.max_recognition_output_tokens}[phase]
        return {"num_ctx": self.context_tokens, "temperature": self.temperature,
                "top_p": self.top_p, "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty, "num_predict": limit,
                "seed": seed}


@dataclass(frozen=True)
class ChatResult:
    content: str
    request: dict
    response: dict
    elapsed_seconds: float


class OllamaClient:
    @classmethod
    def from_config(cls, config, model, **kwargs):
        if config.get("runtime") != "ollama":
            raise ValueError("Only the agreed local Ollama runtime is supported")
        selected = [item for item in config["models"] if item["tag"] == model]
        if len(selected) != 1:
            raise ValueError("Select exactly one of the configured models")
        thinking = selected[0].get("supports_thinking")
        if type(thinking) is not bool:
            raise ValueError("Record the model's measured thinking capability first")
        return cls(model, settings=GenerationSettings.from_config(config["generation"]),
                   supports_thinking=thinking, **kwargs)

    def __init__(self, model, base_url="http://127.0.0.1:11434", settings=None,
                 timeout=180, transport=None, supports_thinking=False):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query
                or parsed.fragment or parsed.path not in {"", "/"}):
            raise ValueError("Only a local HTTP Ollama endpoint is permitted")
        if not model or ":cloud" in model:
            raise ValueError("A local model tag is required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.settings = settings or GenerationSettings()
        self.timeout = timeout
        self.supports_thinking = supports_thinking
        self._transport = transport
        self._lock = Lock()
        self._opener = build_opener(ProxyHandler({}), _NoRedirects())

    def _request(self, method, path, payload=None):
        if self._transport is not None:
            return self._transport(method, path, payload)
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(self.base_url + path, data=body, method=method,
                          headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if payload is not None and payload.get("stream") is True:
                    return read_chat_stream(response)
                return json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 500 and payload is not None and payload.get("stream") is True:
                try:
                    provider_error = json.loads(detail)
                except json.JSONDecodeError:
                    provider_error = None
                if isinstance(provider_error, dict) and provider_error.get("error") == REPETITION_ERROR:
                    return aborted_response("", [provider_error], http_status=500)
            exception = ContextCapacityError if (
                error.code == 400 and any(word in detail.lower()
                                         for word in ("context", "exceed", "input length"))
            ) else OllamaError
            raise exception(f"Ollama HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise OllamaError(f"Local Ollama request failed: {error}") from error

    def inventory(self):
        return {"version": self._request("GET", "/api/version"),
                "models": self._request("GET", "/api/tags")}

    def show(self):
        return self._request("POST", "/api/show", {"model": self.model})

    def running(self):
        return self._request("GET", "/api/ps")

    def unload(self):
        return self._request("POST", "/api/chat",
                             {"model": self.model, "messages": [],
                              "stream": False, "keep_alive": 0})

    def prepare_correction(self, result):
        if not generation_aborted(result.response):
            return None
        # The same request works from a fresh load but can repeat after prior requests.
        # Reset only after this definite generation failure, before the single correction.
        with self._lock:
            response = self.unload()
        if response.get("done") is not True or response.get("done_reason") != "unload":
            raise OllamaError("Model state could not be reset before correction")
        return {"kind": "model_unload_before_correction", "model": self.model, "response": response}

    def build_payload(self, messages, schema, phase, seed):
        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema,
            "stream": True,
            "truncate": False,
            "shift": False,
            "keep_alive": "5m",
            "options": self.settings.options(phase, seed),
        }
        if self.supports_thinking:
            payload["think"] = False
        return payload

    def chat(self, messages, schema, phase, seed):
        payload = self.build_payload(messages, schema, phase, seed)
        # No hidden chat state or parallel inference requests are retained.
        with self._lock:
            started = perf_counter()
            response = self._request("POST", "/api/chat", payload)
            elapsed = perf_counter() - started
        if generation_aborted(response):
            return ChatResult(response["message"]["content"], payload, response, elapsed)
        if not isinstance(response, dict) or response.get("done") is not True:
            raise OllamaError("Ollama did not return a completed response")
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaError("Ollama response has no text content")
        if message.get("thinking"):
            raise OllamaError("Unexpected thinking output; check model mode")
        return ChatResult(message["content"], payload, response, elapsed)

