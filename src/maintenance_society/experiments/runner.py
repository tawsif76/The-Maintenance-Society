"""Durable episode execution. Completed rounds are replayed locally without inference."""

from dataclasses import asdict
from pathlib import Path

from maintenance_society.agents.controller import AgentController, EpisodeIncomplete
from maintenance_society.agents.observations import (
    DiscussionRound, ObservationBuilder, ProposalEntry, snapshot,
)
from maintenance_society.agents.schemas import Proposal, parse_response
from maintenance_society.agents.ollama_client import OllamaError, ContextCapacityError
from maintenance_society.environment.workshop import Workshop
from maintenance_society.evaluation.agent_metrics import agent_metrics
from .agent_round import AgentRoundCoordinator
from .storage import JournalClient, atomic_json, canonical, digest, read_json, utc_now


def discussion_document(discussion):
    if discussion is None:
        return None
    return {"round_number": discussion.round_number, "entries": [
        {"agent_id": entry.agent_id, "proposal": entry.proposal.model_dump()}
        for entry in discussion.entries]}


def load_discussion(value):
    if value is None:
        return None
    return DiscussionRound(value["round_number"], tuple(
        ProposalEntry(entry["agent_id"], Proposal.model_validate(entry["proposal"], strict=True))
        for entry in value["entries"]))


def execute_episode(config, spec, client, prompts_dir, directory, progress=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "episode.json", {"spec": spec.to_dict(), "workshop": asdict(config)}, immutable=True)
    result_path = directory / "result.json"
    if result_path.exists():
        return read_json(result_path)
    workshop = Workshop(config)
    builder = ObservationBuilder(prompts_dir)
    journal = JournalClient(client, directory / "requests")
    controller = AgentController(journal)
    coordinator = AgentRoundCoordinator(builder, controller)
    discussions = []
    recognition = {}
    round_dir = directory / "rounds"
    round_dir.mkdir(exist_ok=True)
    existing = sorted(round_dir.glob("*.json"))
    if [path.name for path in existing] != [f"{number:03d}.json" for number in range(1, len(existing) + 1)]:
        raise ValueError("Saved rounds must be a contiguous prefix")
    if len(existing) > config.rounds_per_episode:
        raise ValueError("Too many saved rounds")

    def checkpoint(status, error=None):
        atomic_json(directory / "checkpoint.json", {
            "episode_id": spec.episode_id, "status": status, "updated_at_utc": utc_now(),
            "completed_rounds": workshop.state.completed_rounds,
            "recognition_evaluators_completed": sorted(recognition),
            "state": asdict(workshop.state), "error": error,
        })

    status = "complete"
    error_info = None
    halt_batch = False
    checkpoint("running")
    try:
        while not workshop.done:
            number = workshop.state.next_round
            path = round_dir / f"{number:03d}.json"
            if path.exists():
                saved = read_json(path)
                if saved["episode_id"] != spec.episode_id:
                    raise ValueError("Saved round belongs to another episode")
                discussion = load_discussion(saved["discussion"])
                if workshop.state.shutdown_remaining:
                    if discussion is not None:
                        raise ValueError("A shutdown cannot have proposals")
                    record = workshop.step()
                else:
                    if spec.condition in "CD":
                        if discussion is None or discussion.round_number != number:
                            raise ValueError("Saved communication round has no matching discussion")
                        builder.validate_discussion(discussion, snapshot(workshop, spec.display_order))
                    elif discussion is not None:
                        raise ValueError("Unexpected discussion")
                    record = workshop.step(saved["record"]["actions"])
                if record.to_dict(config.agent_ids) != saved["record"]:
                    raise ValueError("Saved round disagrees with simulator replay")
            else:
                completed = coordinator.run(workshop, spec.condition, spec.seed,
                                            tuple(discussions), spec.display_order)
                record, discussion = completed.record, completed.discussion
                atomic_json(path, {
                    "episode_id": spec.episode_id, "record": record.to_dict(config.agent_ids),
                    "discussion": discussion_document(discussion),
                    "request_hashes": [digest(attempt.result.request) for attempt in completed.attempts],
                }, immutable=True)
            if discussion is not None:
                discussions.append(discussion)
            checkpoint("running")
            if progress:
                progress(f"{spec.episode_id}: round {number}/{config.rounds_per_episode}")

        for identity in spec.display_order:
            path = directory / "recognition" / f"{identity}.json"
            if path.exists():
                value = read_json(path)
                if value["agent_id"] != identity:
                    raise ValueError("Recognition evaluator mismatch")
                validated = parse_response(canonical({"credits": value["credits"]}),
                                           "recognition", identity, config.agent_ids)
            else:
                observation = builder.build_recognition(workshop, identity, spec.condition,
                                                        tuple(discussions), spec.display_order)
                validated = controller.request(observation, spec.seed).value
                atomic_json(path, {"agent_id": identity, "credits": validated.credits}, immutable=True)
            recognition[identity] = dict(validated.credits)
            checkpoint("recognition")
    except KeyboardInterrupt:
        checkpoint("interrupted", "Stopped by user; resume from saved records")
        raise
    except (EpisodeIncomplete, OllamaError) as error:
        causes = []
        underlying = error
        while underlying is not None:
            causes.append(underlying)
            underlying = underlying.__cause__
        infrastructure = any(isinstance(item, OllamaError) for item in causes)
        context = any(isinstance(item, ContextCapacityError) for item in causes)
        status = "incomplete"
        halt_batch = infrastructure
        error_info = {"kind": "context_capacity" if context else (
            "infrastructure_or_uncertain_request" if infrastructure else "invalid_output"),
            "message": str(error)}
    except BaseException as error:
        checkpoint("interrupted", f"{type(error).__name__}: {error}")
        raise

    result = {
        "episode_id": spec.episode_id, "model": spec.model, "condition": spec.condition,
        "implementation_fingerprint": getattr(client, "implementation_fingerprint", None),
        "episode_index": spec.episode_index, "seed": spec.seed, "display_order": list(spec.display_order),
        "status": status, "workshop_complete": workshop.done,
        "completed_rounds": workshop.state.completed_rounds,
        "error": error_info, "halt_batch": halt_batch,
        "partial_state": asdict(workshop.state),
        "metrics": agent_metrics(workshop, spec.condition, discussions, recognition),
        "requests": journal.statistics(), "finished_at_utc": utc_now(),
    }
    atomic_json(result_path, result, immutable=True)
    checkpoint(status, error_info)
    return read_json(result_path)
