"""Validate one agent's output. Invalid content receives exactly one correction."""

from dataclasses import dataclass
import hashlib
import json

from pydantic import ValidationError

from .ollama_client import ChatResult, generation_aborted
from .schemas import Decision, Proposal, Recognition, parse_response, response_schema


@dataclass(frozen=True)
class Attempt:
    agent_id: str
    round_number: int
    condition: str
    phase: str
    attempt_number: int
    valid: bool
    error: str | None
    result: ChatResult
    error_kind: str | None = None


@dataclass(frozen=True)
class AgentReply:
    agent_id: str
    value: Decision | Proposal | Recognition
    attempts: tuple[Attempt, ...]


class EpisodeIncomplete(RuntimeError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = tuple(attempts)


def request_seed(episode_seed, agent_id, round_number, phase, attempt=0):
    text = f"{episode_seed}|{agent_id}|{round_number}|{phase}|{attempt}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % (2**31)


class AgentController:
    def __init__(self, client):
        self.client = client

    def request(self, observation, episode_seed):
        messages = observation.messages()
        attempts = []
        for index in range(2):
            result = self.client.chat(
                messages=messages,
                schema=response_schema(observation.phase, observation.agent_id, observation.agent_ids),
                phase=observation.phase,
                seed=request_seed(episode_seed, observation.agent_id,
                                  observation.round_number, observation.phase, index),
            )
            error = None
            error_kind = None
            value = None
            try:
                if generation_aborted(result.response):
                    raise ValueError("Model output repeated until generation was aborted. "
                                     "Return a complete JSON object within the original limits.")
                if result.response.get("done_reason") != "stop":
                    raise ValueError("Response stopped without a normal stop signal")
                value = parse_response(result.content, observation.phase,
                                       observation.agent_id, observation.agent_ids)
            except (ValueError, ValidationError) as invalid:
                # ValidationError is included explicitly to document the contract.
                error = str(invalid)
                error_kind = (
                    "generation_repetition" if generation_aborted(result.response) else
                    "generation_stop" if result.response.get("done_reason") != "stop" else
                    "recognition_total" if "sum to exactly 100" in error else
                    "json_format" if isinstance(invalid, json.JSONDecodeError) else "schema_or_constraint")
            attempts.append(Attempt(
                observation.agent_id, observation.round_number, observation.condition,
                observation.phase, index + 1, error is None, error, result, error_kind,
            ))
            recorder = getattr(self.client, "record_attempt", None)
            if recorder is not None:
                recorder(attempts[-1])
            if error is None:
                return AgentReply(observation.agent_id, value, tuple(attempts))
            if index == 0:
                prepare = getattr(self.client, "prepare_correction", None)
                if prepare is not None:
                    prepare(result)
                if generation_aborted(result.response):
                    # Echoing a long repeated fragment primed Phi to repeat it again,
                    # including after a fresh load. Keep that fragment in the journal,
                    # and provide failure feedback without putting it back in context.
                    messages = observation.messages() + [{
                        "role": "user", "content": "Your previous generation repeated tokens and was aborted. "
                        "Start a fresh, concise response. Do not repeat words or action codes. "
                        "Return exactly one complete JSON object matching the original schema.",
                    }]
                else:
                    messages = messages + [
                        {"role": "assistant", "content": result.content},
                        {"role": "user",
                         "content": "Your response was invalid. Correct only its format or "
                                    "invalid fields. Return one JSON object matching the "
                                    "original schema. Validation error: " + error},
                    ]
        raise EpisodeIncomplete(
            f"{observation.agent_id} returned two invalid {observation.phase} responses",
            attempts,
        )

