"""Public snapshots prevent current-round decisions from leaking between calls."""

from dataclasses import dataclass
import json
from pathlib import Path

from maintenance_society.environment.workshop import RoundRecord, Workshop, WorkshopConfig
from .schemas import Proposal, response_schema

CONDITIONS = {
    "A": (False, False), "B": (True, False),
    "C": (False, True), "D": (True, True),
}
PROMPT_VERSION = "workshop-agent-v2"


@dataclass(frozen=True)
class ProposalEntry:
    agent_id: str
    proposal: Proposal


@dataclass(frozen=True)
class DiscussionRound:
    round_number: int
    entries: tuple[ProposalEntry, ...]


@dataclass(frozen=True)
class PublicSnapshot:
    round_number: int
    wear: int
    config: WorkshopConfig
    agent_ids: tuple[str, ...]
    display_order: tuple[str, ...]
    scores: tuple[int, ...]
    maintenance_counts: tuple[int, ...]
    history: tuple[RoundRecord, ...]


@dataclass(frozen=True)
class AgentObservation:
    agent_id: str
    condition: str
    phase: str
    round_number: int
    system_text: str
    user_text: str
    agent_ids: tuple[str, ...] = ()

    def messages(self):
        # Each caller receives new dictionaries. No shared chat session exists.
        return [{"role": "system", "content": self.system_text},
                {"role": "user", "content": self.user_text}]


def snapshot(workshop: Workshop, display_order=None):
    if workshop.done or workshop.state.shutdown_remaining:
        raise ValueError("Agent observations require an operational round")
    ids = workshop.config.agent_ids
    order = ids if display_order is None else tuple(display_order)
    if len(order) != len(ids) or set(order) != set(ids):
        raise ValueError("Display order must contain every agent exactly once")
    state = workshop.state
    return PublicSnapshot(
        state.next_round, state.wear, workshop.config, ids, order,
        state.scores, state.maintenance_counts, workshop.history,
    )


class ObservationBuilder:
    def __init__(self, prompts_dir: str | Path):
        directory = Path(prompts_dir)
        self.templates = {
            phase: (directory / f"{phase}.txt").read_text(encoding="utf-8").strip()
            for phase in ("system", "decision", "proposal", "recognition")
        }

    @staticmethod
    def validate_discussion(discussion, public):
        if not isinstance(discussion, DiscussionRound):
            raise ValueError("Expected a validated discussion round")
        ids = [entry.agent_id for entry in discussion.entries]
        if len(ids) != len(public.agent_ids) or set(ids) != set(public.agent_ids):
            raise ValueError("A discussion requires exactly one proposal per agent")
        if any(not isinstance(entry.proposal, Proposal) for entry in discussion.entries):
            raise ValueError("Unvalidated proposals cannot enter observations")

    def build(self, public, agent_id, condition, phase, past_discussions=(),
              current_proposals=None):
        if agent_id not in public.agent_ids:
            raise ValueError("Unknown agent")
        if condition not in CONDITIONS or phase not in {"proposal", "decision"}:
            raise ValueError("Unknown condition or phase")
        maintenance_visible, communication = CONDITIONS[condition]
        if phase == "proposal" and not communication:
            raise ValueError("This condition has no proposal phase")
        discussions = tuple(past_discussions)
        numbers = [item.round_number for item in discussions]
        if numbers != sorted(set(numbers)) or any(
            number < 1 or number >= public.round_number for number in numbers
        ):
            raise ValueError("Past discussions must be unique, ordered, and earlier")
        for discussion in discussions:
            self.validate_discussion(discussion, public)
        if not communication and (discussions or current_proposals is not None):
            raise ValueError("Messages cannot enter a non-communication condition")
        if phase == "proposal" and current_proposals is not None:
            raise ValueError("Proposers cannot see current-round proposals")
        if phase == "decision" and communication:
            if current_proposals is None:
                raise ValueError("All current proposals are required before decisions")
            self.validate_discussion(current_proposals, public)
            if current_proposals.round_number != public.round_number:
                raise ValueError("Current proposals belong to another round")
            discussions += (current_proposals,)
        discussions = discussions[-5:]

        return self._render(public, agent_id, condition, phase, maintenance_visible, discussions)

    def build_recognition(self, workshop, agent_id, condition, past_discussions=(),
                          display_order=None):
        if not workshop.done or condition not in CONDITIONS:
            raise ValueError("Recognition requires a completed workshop and valid condition")
        ids = workshop.config.agent_ids
        order = ids if display_order is None else tuple(display_order)
        if agent_id not in ids or len(order) != len(ids) or set(order) != set(ids):
            raise ValueError("Invalid evaluator or display order")
        public = PublicSnapshot(workshop.state.completed_rounds, workshop.state.wear,
                                workshop.config, ids, order, workshop.state.scores,
                                workshop.state.maintenance_counts, workshop.history)
        discussions = tuple(past_discussions)
        numbers = [item.round_number for item in discussions]
        if numbers != sorted(set(numbers)) or any(n < 1 or n > 30 for n in numbers):
            raise ValueError("Recognition discussions must be unique and ordered")
        if not CONDITIONS[condition][1] and discussions:
            raise ValueError("This condition has no discussion")
        for discussion in discussions:
            self.validate_discussion(discussion, public)
        return self._render(public, agent_id, condition, "recognition", CONDITIONS[condition][0],
                            discussions[-5:])

    def _render(self, public, agent_id, condition, phase, maintenance_visible, discussions):
        scores = dict(zip(public.agent_ids, public.scores))
        counts = dict(zip(public.agent_ids, public.maintenance_counts))
        dashboard = []
        for identity in public.display_order:
            row = {"agent": identity, "production_points": scores[identity]}
            if maintenance_visible:
                row["maintenance_turns"] = counts[identity]
            dashboard.append(row)
        history = []
        for record in public.history:
            actions = dict(zip(public.agent_ids, record.actions))
            codes = "".join({"produce": "P", "maintain": "M"}.get(
                actions.get(identity), "-") for identity in public.display_order)
            history.append([
                record.round_number, record.starting_wear, record.ending_wear,
                codes, record.failure_event, record.automatic_repair,
            ])
        messages = []
        for discussion in discussions:
            by_id = {entry.agent_id: entry.proposal for entry in discussion.entries}
            messages.append({
                "round": discussion.round_number,
                "proposals": [{"agent": identity, **by_id[identity].model_dump()}
                              for identity in public.display_order],
            })
        data = {
            "you": agent_id,
            "round": public.round_number,
            "total_rounds": public.config.rounds_per_episode,
            "rounds_after_this_one": public.config.rounds_per_episode - public.round_number,
            "starting_wear": public.wear,
            "agent_order": list(public.display_order),
            "dashboard": dashboard,
            "history_columns": ["round", "starting_wear", "ending_wear",
                                "actions_in_agent_order", "failure", "repair"],
            "history_action_codes": {"P": "produce", "M": "maintain", "-": "shutdown"},
            "complete_action_history": history,
            "recent_discussions": messages,
        }
        if phase == "recognition":
            data.pop("round")
            data.pop("rounds_after_this_one")
            data["completed_rounds"] = public.round_number
            data["episode_complete"] = True
        cfg = public.config
        system = self.templates["system"].format(
            agents=cfg.agents, rounds=cfg.rounds_per_episode,
            points=cfg.production_points, production_wear=cfg.production_wear,
            maintenance_capacity=cfg.maintenance_capacity,
            initial_wear=cfg.initial_wear, failure_threshold=cfg.failure_threshold,
            shutdown_rounds=cfg.shutdown_rounds, repair_wear=cfg.repair_wear,
        )
        instruction = self.templates[phase].format(
            schema=json.dumps(response_schema(phase, agent_id, public.display_order), separators=(",", ":"))
        )
        return AgentObservation(
            agent_id, condition, phase, public.round_number, system,
            json.dumps(data, separators=(",", ":"), ensure_ascii=False)
            + "\n\n" + instruction,
            public.agent_ids,
        )

