"""Five reference policies. These are ordinary rules, not LLM agents."""

from dataclasses import dataclass
from typing import Callable

from .workshop import WorkshopConfig, WorkshopState


def all_produce(state, config):
    return dict.fromkeys(config.agent_ids, "produce")


def _one_maintainer(config, index):
    actions = dict.fromkeys(config.agent_ids, "produce")
    actions[config.agent_ids[index]] = "maintain"
    return actions


def fixed_rotation(state, config):
    return _one_maintainer(config, state.completed_rounds % config.agents)


def designated_threshold(state, config):
    if state.wear >= 24:
        return _one_maintainer(config, 0)
    return all_produce(state, config)


def equitable_threshold(state, config):
    if state.wear >= 24:
        # Advance the rotation only when a maintenance event actually occurs.
        index = sum(state.maintenance_counts) % config.agents
        return _one_maintainer(config, index)
    return all_produce(state, config)


def terminal_aware_threshold(state, config):
    if state.next_round == config.rounds_per_episode:
        return all_produce(state, config)
    return designated_threshold(state, config)


@dataclass(frozen=True)
class ReferencePolicy:
    name: str
    description: str
    choose: Callable[[WorkshopState, WorkshopConfig], dict[str, str]]


REFERENCE_POLICIES = (
    ReferencePolicy("always_produce", "All agents produce whenever operational",
                    all_produce),
    ReferencePolicy("fixed_rotation", "One rotating maintainer every round",
                    fixed_rotation),
    ReferencePolicy("designated_threshold", "Agent 1 maintains at starting wear >= 24",
                    designated_threshold),
    ReferencePolicy("equitable_threshold", "Rotate responsibility at each threshold event",
                    equitable_threshold),
    ReferencePolicy("terminal_aware_threshold", "Designated threshold; all produce in round 30",
                    terminal_aware_threshold),
)

