"""Metrics derived from recorded workshop state. No model judgments."""

from dataclasses import asdict, dataclass

from maintenance_society.environment.workshop import Workshop


@dataclass(frozen=True)
class EpisodeSummary:
    policy: str
    completed_rounds: int
    total_items: int
    group_points: int
    maintenance_turns: int
    maintenance_concentration: float | None
    score_gap: int
    failure_events: int
    lost_rounds: int
    wasted_wear_reduction: int
    final_wear: int
    scores: dict[str, int]
    maintenance_counts: dict[str, int]

    def to_dict(self):
        return asdict(self)


def summarise(workshop: Workshop, policy: str) -> EpisodeSummary:
    """Summarise a completed episode; zero maintenance has no concentration."""
    if not workshop.done:
        raise ValueError("A completed episode is required for its final summary")
    state = workshop.state
    total_maintenance = sum(state.maintenance_counts)
    return EpisodeSummary(
        policy=policy,
        completed_rounds=state.completed_rounds,
        total_items=sum(record.producers for record in workshop.history),
        group_points=sum(state.scores),
        maintenance_turns=total_maintenance,
        maintenance_concentration=(max(state.maintenance_counts) / total_maintenance
                                   if total_maintenance else None),
        score_gap=max(state.scores) - min(state.scores),
        failure_events=state.failure_events,
        lost_rounds=state.lost_rounds,
        wasted_wear_reduction=state.wasted_wear_reduction,
        final_wear=state.wear,
        scores=dict(zip(workshop.config.agent_ids, state.scores)),
        maintenance_counts=dict(zip(workshop.config.agent_ids, state.maintenance_counts)),
    )

