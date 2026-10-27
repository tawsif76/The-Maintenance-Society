"""Matched seeds and display orders for the fixed 80-episode main experiment."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import random
import sys

import pydantic

from maintenance_society.environment.workshop import WorkshopConfig
from maintenance_society.agents.observations import CONDITIONS, PROMPT_VERSION
from maintenance_society.agents.ollama_client import GenerationSettings
from .storage import digest, read_json

TAGS = ("qwen3:4b-instruct-2507-q4_K_M", "phi4-mini:3.8b-q4_K_M")


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    model: str
    condition: str
    episode_index: int
    seed: int
    display_order: tuple[str, ...]

    def to_dict(self):
        return {**asdict(self), "display_order": list(self.display_order)}


def validate_design(experiment, models):
    rules = {**asdict(WorkshopConfig()), "simultaneous_actions": True,
             "score_before_failure_check": True, "record_final_round_failure": True}
    if experiment["workshop"] != rules:
        raise ValueError("The runner requires the agreed workshop rules")
    expected = {name: {"maintenance_totals_in_dashboard": flags[0], "communication": flags[1]}
                for name, flags in CONDITIONS.items()}
    if experiment["conditions"] != expected:
        raise ValueError("The four conditions differ from the implemented design")
    schedule = experiment["schedule"]
    if (schedule["total_main_episodes"] != 80 or schedule["model_count"] != 2
            or schedule["episodes_per_model_per_condition"] != 10
            or schedule["episode_ids"] != list(range(10))
            or schedule["pilot"] is not False or schedule["parameter_sensitivity_study"] is not False):
        raise ValueError("The runner requires the agreed 80-episode schedule")
    expected_protocol = {
        "type": "fixed turn-based", "actions": ["produce", "maintain"],
        "proposal_collection": "independent; reveal together",
        "final_decision_collection": "independent; reveal only after all commit",
        "max_message_words": 60, "discussion_rounds_in_context": 5,
        "action_history": "complete", "silent_truncation": False,
        "max_model_correction_requests": 1, "on_second_invalid_output": "mark_episode_incomplete",
        "model_calls_during_shutdown": False,
    }
    if experiment["protocol"] != expected_protocol:
        raise ValueError("Protocol differs from this implementation")
    if experiment["evaluation"] != {
        "end_of_episode_recognition": True, "recognition_points_per_evaluator": 100,
        "self_credit_allowed": False, "recognition_changes_rewards": False,
        "final_rounds_to_report_separately": 6, "zero_maintenance_concentration": None,
        "zero_declaration_fulfilment": None,
    }:
        raise ValueError("Evaluation settings differ from this implementation")
    if experiment["scope"] != {"local_only": True, "cloud_inference": False, "training": False}:
        raise ValueError("Only the local inference design is supported")
    if tuple(item["tag"] for item in models["models"]) != TAGS or models["runtime"] != "ollama":
        raise ValueError("Use the two agreed local models")
    if models["execution"] != {"max_loaded_models": 1, "max_in_flight_requests": 1,
                               "separate_agent_observations": True}:
        raise ValueError("This runner supports one loaded model and one request at a time")
    if models["reproducibility"]["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Prompt version does not match the implementation")
    for model in models["models"]:
        value = model.get("digest")
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                or type(model.get("supports_thinking")) is not bool):
            raise ValueError("Record model digests and capabilities before execution")
    GenerationSettings.from_config(models["generation"])


def make_schedule(base_seed=20261023):
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("Base seed must be a nonnegative integer")
    result = []
    for model_index, model in enumerate(TAGS):
        for index in range(10):
            seed = int.from_bytes(hashlib.sha256(f"{base_seed}|{index}".encode()).digest()[:4], "big") % 2**31
            order = list(WorkshopConfig().agent_ids)
            random.Random(seed).shuffle(order)
            conditions = list(CONDITIONS)
            random.Random(seed + 1).shuffle(conditions)
            for condition in conditions:
                result.append(EpisodeSpec(f"m{model_index}-{condition}-{index:02d}", model,
                                          condition, index, seed, tuple(order)))
    return tuple(result)


def fingerprint(root):
    root = Path(root)
    paths = list((root / "src").rglob("*.py")) + list((root / "prompts").glob("*.txt"))
    paths += [root / "configs/experiment.json", root / "configs/models.json",
              root / "requirements.txt", root / "scripts/run_experiment.py"]
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(paths)}
    value = {"files": hashes, "python": sys.version, "pydantic": pydantic.__version__}
    return {"sha256": digest(value), **value}
