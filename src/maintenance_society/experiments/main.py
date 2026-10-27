"""Freeze the main study and execute or inspect its matched schedule."""

import csv
import io
from pathlib import Path

from maintenance_society.agents.ollama_client import OllamaClient, OllamaError
from maintenance_society.environment.workshop import load_config
from .runner import execute_episode
from .schedule import fingerprint, make_schedule, validate_design
from .storage import RunLock, atomic_json, read_json, utc_now


def prepare_run(root, directory, base_seed):
    root, directory = Path(root), Path(directory)
    experiment = read_json(root / "configs/experiment.json")
    models = read_json(root / "configs/models.json")
    validate_design(experiment, models)
    code = fingerprint(root)
    schedule = make_schedule(base_seed)
    specification = {"run_kind": "main_experiment", "base_seed": base_seed,
                     "fingerprint": code, "experiment": experiment, "models": models,
                     "schedule": [spec.to_dict() for spec in schedule]}
    path = directory / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        actual = {key: value for key, value in manifest.items() if key != "created_at_utc"}
        if actual != specification:
            raise ValueError("This run belongs to a different code or configuration version. "
                             "Use --status to inspect it, or a new output directory for a new run.")
    else:
        unexpected = [item for item in directory.iterdir() if item.name != "run.lock"] if directory.exists() else []
        if unexpected:
            raise ValueError("A new run directory must be empty")
        atomic_json(path, {"created_at_utc": utc_now(), **specification}, immutable=True)
    return schedule, models, code


def export_status(directory, write=True):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    rows = []
    for spec in manifest["schedule"]:
        episode_dir = directory / "episodes" / spec["episode_id"]
        path = episode_dir / "result.json"
        result = read_json(path) if path.exists() else None
        checkpoint = read_json(episode_dir / "checkpoint.json") if (episode_dir / "checkpoint.json").exists() else {}
        status = result["status"] if result else ("unfinished" if checkpoint else "pending")
        metrics = (result.get("metrics") or {}) if result else {}
        requests = result["requests"] if result else {}
        row = {
            "episode_id": spec["episode_id"], "model": spec["model"], "condition": spec["condition"],
            "episode_index": spec["episode_index"], "seed": spec["seed"], "status": status,
            "workshop_complete": result["workshop_complete"] if result else False,
            "completed_rounds": result["completed_rounds"] if result else checkpoint.get("completed_rounds", 0),
            "error_kind": result["error"]["kind"] if result and result["error"] else None,
            "implementation_fingerprint": (result.get("implementation_fingerprint")
                                             or manifest["fingerprint"]["sha256"]) if result else None,
        }
        for name in ("group_points", "total_items", "maintenance_turns", "maintenance_concentration",
                     "score_gap", "failure_events", "lost_rounds", "wasted_wear_reduction", "final_wear"):
            row[name] = metrics.get(name)
        row["commitment_fulfilment_rate"] = metrics.get("commitments", {}).get("fulfilment_rate")
        for name in ("requests_started", "responses_saved", "requests_without_saved_response",
                     "invalid_responses", "invalid_response_rate", "correction_requests",
                     "prompt_eval_count", "eval_count", "elapsed_inference_seconds",
                     "generation_repetition_responses", "responses_without_token_usage",
                     "model_resets_before_correction"):
            row[name] = requests.get(name)
        rows.append(row)
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("complete", "incomplete", "unfinished", "pending")}
    report = {
        "run_kind": "main_experiment", "updated_at_utc": utc_now(), "planned_episodes": len(rows),
        **counts, "episode_completion_rate_of_planned": counts["complete"] / len(rows),
        "workshop_completion_rate_of_planned": sum(row["workshop_complete"] for row in rows) / len(rows),
        "note": "Every planned episode is listed. Incomplete episodes are retained, not replaced. "
                "Physical metrics can exist when recognition failed; use status when selecting complete episodes.",
        "episodes": rows,
    }
    if not write:
        return report
    atomic_json(directory / "summary.json", report)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temporary = directory / "summary.csv.tmp"
    temporary.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    temporary.replace(directory / "summary.csv")
    return report


def _verify_runtime(client, models):
    """Require the configured runtime and exact model before main-episode inference."""
    inventory = client.inventory()
    if inventory["version"].get("version") != models["reproducibility"]["ollama_version"]:
        raise OllamaError("Ollama version differs from configs/models.json")
    selected = next(item for item in models["models"] if item["tag"] == client.model)
    installed = next((item for item in inventory["models"].get("models", [])
                      if item.get("name") == client.model), None)
    if installed is None or installed.get("digest") != selected["digest"]:
        raise OllamaError("The installed model digest differs from configs/models.json")
    others = [item.get("name", item.get("model")) for item in client.running().get("models", [])
              if item.get("digest") != selected["digest"]]
    if others:
        raise OllamaError("Unload other models before this run: " + ", ".join(others))
    capabilities = client.show().get("capabilities", [])
    if ("thinking" in capabilities) != selected["supports_thinking"]:
        raise OllamaError("The model's thinking capability differs from the configuration")


def run_main(root, directory, base_seed=20261023, progress=print):
    root, directory = Path(root), Path(directory)
    # Finished runs remain readable after code cleanup and are never executed again.
    if (directory / "manifest.json").exists():
        report = export_status(directory, write=False)
        if report["pending"] == 0 and report["unfinished"] == 0:
            progress(f"All {report['planned_episodes']} scheduled attempts have ended: "
                     f"{report['complete']} complete, {report['incomplete']} incomplete.")
            progress("No model requests made. Saved experiment records are unchanged.")
            return report
    # Keep the project-wide inference lock outside the checkpoint collection.
    with RunLock(root / "data"), RunLock(directory):
        schedule, models, code = prepare_run(root, directory, base_seed)
        export_status(directory)
        config = load_config(root / "configs/experiment.json")
        try:
            for selected in models["models"]:
                pending = [spec for spec in schedule if spec.model == selected["tag"]
                           and not (directory / "episodes" / spec.episode_id / "result.json").exists()]
                if not pending:
                    continue
                client = OllamaClient.from_config(models, selected["tag"])
                client.implementation_fingerprint = code["sha256"]
                try:
                    _verify_runtime(client, models)
                    for spec in pending:
                        progress(f"Starting {spec.episode_id}: {spec.model}, condition {spec.condition}")
                        result = execute_episode(config, spec, client, root / "prompts",
                                                 directory / "episodes" / spec.episode_id, progress=progress)
                        report = export_status(directory)
                        progress(f"Progress: {report['complete']} complete, {report['incomplete']} incomplete, "
                                 f"{report['pending']} pending, {report['unfinished']} unfinished")
                        if result["halt_batch"]:
                            progress("Batch stopped after an infrastructure/context problem. Records are preserved.")
                            return report
                finally:
                    client.unload()
        finally:
            # Ctrl+C and unexpected errors also leave an up-to-date, derived status table.
            export_status(directory)
        return export_status(directory)
