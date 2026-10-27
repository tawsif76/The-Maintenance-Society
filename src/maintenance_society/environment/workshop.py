"""Deterministic workshop rules. No model calls belong in this module."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class WorkshopConfig:
    agents: int = 6
    rounds_per_episode: int = 30
    initial_wear: int = 12
    failure_threshold: int = 30
    shutdown_rounds: int = 2
    repair_wear: int = 12
    production_points: int = 3
    production_wear: int = 1
    maintenance_capacity: int = 6
    wear_floor: int = 0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int:
                raise ValueError(f"{field.name} must be an integer")
            if value < 0:
                raise ValueError(f"{field.name} must not be negative")
        positive = ("agents", "rounds_per_episode", "failure_threshold",
                    "shutdown_rounds", "production_points", "production_wear",
                    "maintenance_capacity")
        if any(getattr(self, name) == 0 for name in positive):
            raise ValueError("Counts, thresholds, and action effects must be positive")
        if self.wear_floor != 0:
            raise ValueError("This implementation requires the agreed zero wear floor")
        if self.initial_wear >= self.failure_threshold:
            raise ValueError("Initial wear must be below the failure threshold")
        if self.repair_wear >= self.failure_threshold:
            raise ValueError("Repair wear must be below the failure threshold")

    @property
    def agent_ids(self):
        return tuple(f"agent_{i + 1}" for i in range(self.agents))


def load_config(path: str | Path) -> WorkshopConfig:
    """Read workshop rules from the existing experiment configuration."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = document["workshop"]
    for flag in ("simultaneous_actions", "score_before_failure_check",
                 "record_final_round_failure"):
        if rules.get(flag) is not True:
            raise ValueError(f"The agreed protocol requires {flag}=true")
    names = {field.name for field in fields(WorkshopConfig)}
    allowed = names | {"simultaneous_actions", "score_before_failure_check",
                       "record_final_round_failure"}
    unknown = set(rules) - allowed
    if unknown:
        raise ValueError(f"Unrecognised workshop settings: {sorted(unknown)}")
    return WorkshopConfig(**{name: rules[name] for name in names})


@dataclass(frozen=True)
class WorkshopState:
    completed_rounds: int
    wear: int
    shutdown_remaining: int
    scores: tuple[int, ...]
    maintenance_counts: tuple[int, ...]
    failure_events: int = 0
    lost_rounds: int = 0
    wasted_wear_reduction: int = 0

    @property
    def next_round(self):
        return self.completed_rounds + 1


@dataclass(frozen=True)
class RoundRecord:
    round_number: int
    operational: bool
    starting_wear: int
    ending_wear: int
    actions: tuple[str, ...]
    producers: int
    maintainers: int
    points_awarded: tuple[int, ...]
    wasted_wear_reduction: int
    failure_event: bool
    automatic_repair: bool
    shutdown_before: int
    shutdown_after: int

    def to_dict(self, agent_ids):
        result = asdict(self)
        result["actions"] = dict(zip(agent_ids, self.actions))
        result["points_awarded"] = dict(zip(agent_ids, self.points_awarded))
        return result


class Workshop:
    """Own one episode. Each successful step advances exactly one round."""

    def __init__(self, config: WorkshopConfig):
        self.config = config
        self.reset()

    def reset(self):
        """Start a new episode without retaining previous scores or history."""
        self._state = WorkshopState(
            completed_rounds=0,
            wear=self.config.initial_wear,
            shutdown_remaining=0,
            scores=(0,) * self.config.agents,
            maintenance_counts=(0,) * self.config.agents,
        )
        self._history = []

    @property
    def state(self):
        # Frozen objects and tuples prevent policies from changing the state.
        return self._state

    @property
    def history(self):
        return tuple(self._history)

    @property
    def done(self):
        return self.state.completed_rounds >= self.config.rounds_per_episode

    def step(self, actions: Mapping[str, str] | None = None) -> RoundRecord:
        """Validate a complete joint action, then apply it atomically."""
        if self.done:
            raise RuntimeError("Episode is complete; reset before taking another step")
        old = self.state
        cfg = self.config
        zero_points = (0,) * cfg.agents

        if old.shutdown_remaining:
            if actions is not None and (not isinstance(actions, Mapping) or len(actions)):
                raise ValueError("Shutdown rounds cannot contain agent actions")
            remaining = old.shutdown_remaining - 1
            repaired = remaining == 0
            wear = cfg.repair_wear if repaired else old.wear
            new = replace(
                old, completed_rounds=old.next_round, wear=wear,
                shutdown_remaining=remaining, lost_rounds=old.lost_rounds + 1,
            )
            record = RoundRecord(
                old.next_round, False, old.wear, wear, (), 0, 0, zero_points,
                0, False, repaired, old.shutdown_remaining, remaining,
            )
        else:
            # Validate every action before changing any score or workshop field.
            if not isinstance(actions, Mapping):
                raise ValueError("Operational rounds require one action per agent")
            if set(actions) != set(cfg.agent_ids):
                raise ValueError("Actions must contain exactly the workshop agent IDs")
            ordered = tuple(actions[agent] for agent in cfg.agent_ids)
            if any(type(action) is not str or action not in {"produce", "maintain"}
                   for action in ordered):
                raise ValueError("Each action must be 'produce' or 'maintain'")

            producers = ordered.count("produce")
            maintainers = cfg.agents - producers
            raw_wear = (old.wear + producers * cfg.production_wear
                        - maintainers * cfg.maintenance_capacity)
            wear = max(cfg.wear_floor, raw_wear)
            waste = max(0, cfg.wear_floor - raw_wear)
            failure = wear >= cfg.failure_threshold
            points = tuple(cfg.production_points if action == "produce" else 0
                           for action in ordered)
            new = WorkshopState(
                completed_rounds=old.next_round,
                wear=wear,
                shutdown_remaining=cfg.shutdown_rounds if failure else 0,
                scores=tuple(score + earned for score, earned
                             in zip(old.scores, points)),
                maintenance_counts=tuple(count + (action == "maintain")
                                         for count, action
                                         in zip(old.maintenance_counts, ordered)),
                failure_events=old.failure_events + int(failure),
                lost_rounds=old.lost_rounds,
                wasted_wear_reduction=old.wasted_wear_reduction + waste,
            )
            record = RoundRecord(
                old.next_round, True, old.wear, wear, ordered, producers,
                maintainers, points, waste, failure, False, 0,
                new.shutdown_remaining,
            )

        self._state = new
        self._history.append(record)
        return record

