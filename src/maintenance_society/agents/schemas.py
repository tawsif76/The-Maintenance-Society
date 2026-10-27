"""Strict public response formats. No free-form reasoning trace is requested."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Action = Literal["produce", "maintain"]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    action: Action


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    message: str = Field(min_length=1)
    intended_action: Action

    @field_validator("message")
    @classmethod
    def check_word_limit(cls, value):
        if not value.strip():
            raise ValueError("Message must not be blank")
        if len(value.split()) > 60:
            raise ValueError("Message must contain at most 60 whitespace-separated words")
        return value


class Recognition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    credits: dict[str, int]

    @field_validator("credits")
    @classmethod
    def check_credits(cls, value):
        if any(type(points) is not int or not 0 <= points <= 100 for points in value.values()):
            raise ValueError("Each recognition allocation must be an integer from 0 to 100")
        if sum(value.values()) != 100:
            raise ValueError("Recognition allocations must sum to exactly 100")
        return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-standard JSON constant: {value}")


def parse_response(text, phase, agent_id=None, agent_ids=()):
    """Reject malformed JSON, duplicates, extra fields, and invalid actions."""
    schema = {"decision": Decision, "proposal": Proposal, "recognition": Recognition}[phase]
    value = json.loads(text, object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant)
    result = schema.model_validate(value, strict=True)
    if phase == "recognition":
        peers = set(agent_ids) - {agent_id}
        if agent_id not in agent_ids or len(peers) != 5 or set(result.credits) != peers:
            raise ValueError("Allocate recognition to exactly the other five agents; no self-credit")
    return result


def response_schema(phase, agent_id=None, agent_ids=()):
    schema = {"decision": Decision, "proposal": Proposal, "recognition": Recognition}[phase].model_json_schema()
    if phase == "recognition":
        peers = [identity for identity in agent_ids if identity != agent_id]
        if agent_id not in agent_ids or len(peers) != 5:
            raise ValueError("Recognition requires one evaluator and five peers")
        schema["properties"]["credits"] = {
            "type": "object", "properties": {
                peer: {"type": "integer", "minimum": 0, "maximum": 100} for peer in peers},
            "required": peers, "additionalProperties": False,
        }
    return schema

