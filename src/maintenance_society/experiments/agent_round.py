"""Coordinate a single round. Durable whole-episode recovery is a later layer."""

from dataclasses import dataclass

from maintenance_society.agents.controller import EpisodeIncomplete
from maintenance_society.agents.ollama_client import OllamaError
from maintenance_society.environment.workshop import RoundRecord
from maintenance_society.agents.observations import (
    CONDITIONS, DiscussionRound, ProposalEntry, snapshot,
)


@dataclass(frozen=True)
class CompletedAgentRound:
    record: RoundRecord
    discussion: DiscussionRound | None
    attempts: tuple


class AgentRoundCoordinator:
    def __init__(self, builder, controller):
        self.builder = builder
        self.controller = controller
        self.attempts = []
        self.incomplete = False

    def run(self, workshop, condition, episode_seed, past_discussions=(),
            display_order=None):
        if self.incomplete:
            raise RuntimeError("This episode is incomplete; do not repeat its round")
        if condition not in CONDITIONS:
            raise ValueError("Unknown condition")
        if workshop.done:
            raise RuntimeError("Episode is already complete")
        if workshop.state.shutdown_remaining:
            return CompletedAgentRound(workshop.step(), None, ())

        public = snapshot(workshop, display_order)
        attempts = []
        discussion = None

        def ask(identity, phase):
            observation = self.builder.build(
                public, identity, condition, phase, past_discussions,
                discussion if phase == "decision" else None,
            )
            try:
                reply = self.controller.request(observation, episode_seed)
            except EpisodeIncomplete as error:
                attempts.extend(error.attempts)
                raise
            attempts.extend(reply.attempts)
            return reply.value

        try:
            if CONDITIONS[condition][1]:
                proposals = tuple(
                    ProposalEntry(identity, ask(identity, "proposal"))
                    for identity in public.display_order
                )
                discussion = DiscussionRound(public.round_number, proposals)
            actions = {}
            for identity in public.display_order:
                actions[identity] = ask(identity, "decision").action
        except (EpisodeIncomplete, OllamaError) as error:
            self.incomplete = True
            self.attempts.extend(attempts)
            raise EpisodeIncomplete(
                "Episode incomplete. No actions from this round were applied. "
                + str(error), attempts
            ) from error
        # Applying state only here prevents earlier decisions from leaking.
        record = workshop.step(actions)
        self.attempts.extend(attempts)
        return CompletedAgentRound(record, discussion, tuple(attempts))

