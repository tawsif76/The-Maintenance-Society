"""Aggregate the main experiment's saved results without inference or simulator replay."""

from collections import Counter
from datetime import datetime, timezone
from dataclasses import fields
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics as st
import math

from maintenance_society.environment.policies import REFERENCE_POLICIES
from maintenance_society.environment.workshop import Workshop, WorkshopConfig
from maintenance_society.evaluation.metrics import summarise

BOOTSTRAPS = 20000
METRICS = (
    "group_points", "total_items", "maintenance_turns", "maintenance_concentration",
    "score_gap", "failure_events", "lost_rounds", "wasted_wear_reduction", "final_wear",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(values, label):
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    rng = random.Random(int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big"))
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                     for _ in range(BOOTSTRAPS))

    def quantile(p):
        index = (len(samples) - 1) * p
        low = math.floor(index)
        return samples[low] + (samples[min(low + 1, len(samples) - 1)] - samples[low]) * (index - low)

    return {"n": len(values), "mean": st.mean(values),
            "sd": st.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values), "median": st.median(values),
            "ci95": [quantile(.025), quantile(.975)]}


def spearman(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None

    def ranks(values):
        return [1 + sum(v < x for v in values) + (sum(v == x for v in values) - 1) / 2
                for x in values]

    x, y = ranks(x), ranks(y)
    a, b = st.mean(x), st.mean(y)
    return sum((u - a) * (v - b) for u, v in zip(x, y)) / math.sqrt(
        sum((u - a) ** 2 for u in x) * sum((v - b) ** 2 for v in y))


def terminal_failure(result):
    if not result.get("error"):
        return None
    if result["workshop_complete"]:
        return "recognition_total"
    message = result["error"]["message"]
    if result["error"]["kind"] == "infrastructure_or_uncertain_request":
        return "generation_or_infrastructure"
    return "proposal_word_limit" if "proposal" in message else result["error"]["kind"]


def analyze(run_directory):
    run_directory = Path(run_directory)
    manifest = read_json(run_directory / "manifest.json")
    config = WorkshopConfig(**{f.name: manifest["experiment"]["workshop"][f.name]
                               for f in fields(WorkshopConfig)})
    episodes, rounds = [], []
    transitions = {(mi, c): Counter() for mi in range(2) for c in "CD"}
    for spec in manifest["schedule"]:
        directory = run_directory / "episodes" / spec["episode_id"]
        result = read_json(directory / "result.json")
        mi = int(spec["episode_id"][1])
        row = {"episode_id": spec["episode_id"], "model_index": mi,
               "condition": spec["condition"], "episode_index": spec["episode_index"],
               "seed": spec["seed"], "status": result["status"],
               "workshop_complete": result["workshop_complete"],
               "completed_rounds": result["completed_rounds"],
               "terminal_failure": terminal_failure(result), "metrics": result["metrics"],
               "requests": result["requests"], "partial_state": result["partial_state"]}
        episode_rounds = [read_json(p) for p in sorted((directory / "rounds").glob("*.json"))]
        rounds.extend({"episode_id": spec["episode_id"], "model_index": mi,
                       "condition": spec["condition"], **doc["record"]} for doc in episode_rounds)
        if row["workshop_complete"]:
            metrics = row["metrics"]
            capacity = 6 * metrics["maintenance_turns"]
            row["exploratory"] = {"capacity_waste_fraction":
                                  metrics["wasted_wear_reduction"] / capacity if capacity else None}
            if metrics["recognition"]["complete"]:
                counts = metrics["maintenance_counts"]
                received = metrics["recognition"]["received_totals"]
                row["exploratory"]["recognition_maintenance_spearman"] = spearman(
                    list(counts.values()), [received[agent] for agent in counts])
            if spec["condition"] in "CD":
                counts = transitions[mi, spec["condition"]]
                counts["actual_maintenance"] += metrics["maintenance_turns"]
                for doc in episode_rounds:
                    if doc["discussion"] is None:
                        continue
                    for entry in doc["discussion"]["entries"]:
                        intended = entry["proposal"]["intended_action"]
                        final = doc["record"]["actions"][entry["agent_id"]]
                        counts["declared_maintenance"] += intended == "maintain"
                        counts["maintain_to_maintain"] += intended == final == "maintain"
                        counts["produce_to_maintain"] += intended == "produce" and final == "maintain"
        episodes.append(row)

    cells = []
    for mi, condition in itertools.product(range(2), "ABCD"):
        selected = [r for r in episodes if r["model_index"] == mi and r["condition"] == condition]
        finished = [r for r in selected if r["workshop_complete"]]
        recognized = [r for r in finished if r["metrics"]["recognition"]["complete"]]
        capacity = 6 * sum(r["metrics"]["maintenance_turns"] for r in finished)
        cell = {"model_index": mi, "condition": condition, "scheduled": len(selected),
                "complete": sum(r["status"] == "complete" for r in selected),
                "workshop_complete": len(finished), "recognition_failed": len(finished) - len(recognized),
                "early_failed": len(selected) - len(finished),
                "physical_episode_ids": [r["episode_id"] for r in finished],
                "metrics": {k: describe([r["metrics"][k] for r in finished], f"{mi}{condition}{k}")
                            for k in METRICS},
                "failure_free_episodes": sum(r["metrics"]["failure_events"] == 0 for r in finished),
                "zero_maintenance_episodes": sum(r["metrics"]["maintenance_turns"] == 0 for r in finished),
                "capacity_waste_fraction_pooled":
                    sum(r["metrics"]["wasted_wear_reduction"] for r in finished) / capacity if capacity else None,
                "recognition": {"maintenance_spearman": describe(
                    [r["exploratory"]["recognition_maintenance_spearman"] for r in recognized],
                    f"{mi}{condition}recognition")}}
        if condition in "CD":
            counts = transitions[mi, condition]
            declared = counts["declared_maintenance"]
            cell["commitments"] = {"declarations": declared, "fulfilled": counts["maintain_to_maintain"],
                                   "pooled_rate": counts["maintain_to_maintain"] / declared if declared else None}
        cells.append(cell)

    contrasts = []
    comparisons = [("A", "B", "dashboard_without_discussion"), ("C", "D", "dashboard_with_discussion"),
                   ("A", "C", "discussion_without_dashboard"), ("B", "D", "discussion_with_dashboard")]
    for mi in range(2):
        by = {(r["condition"], r["episode_index"]): r for r in episodes if r["model_index"] == mi}
        for first, second, meaning in comparisons:
            indices = [i for i in range(10) if by[first, i]["workshop_complete"]
                       and by[second, i]["workshop_complete"]]
            metrics = {}
            for name in METRICS:
                differences = [by[second, i]["metrics"][name] - by[first, i]["metrics"][name]
                               for i in indices if by[first, i]["metrics"][name] is not None
                               and by[second, i]["metrics"][name] is not None]
                metrics[name] = describe(differences, f"{mi}{second}-{first}{name}")
            contrasts.append({"model_index": mi, "contrast": f"{second}-{first}", "meaning": meaning,
                              "matched_indices": indices, "n_pairs": len(indices), "metrics": metrics})

    finished = sum(r["workshop_complete"] for r in episodes)
    complete = sum(r["status"] == "complete" for r in episodes)
    analysis = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "analysis_version": "main-analysis-v2",
                "resampling": {"method": "episode percentile bootstrap; differences use matched episode indices",
                               "replicates": BOOTSTRAPS,
                               "seed": "SHA256 of model-condition-metric label, first 8 bytes",
                               "intervals": "descriptive 95%; not adjusted for multiple comparisons"},
                "counts": {"scheduled": len(episodes), "complete": complete, "workshop_complete": finished,
                           "recognition_failed": finished - complete, "early_failed": len(episodes) - finished},
                "terminal_failures": dict(Counter(r["terminal_failure"] for r in episodes if r["terminal_failure"])),
                "cells": cells, "contrasts": contrasts,
                "intention_transitions": [{"model_index": mi, "condition": c, **dict(counts)}
                                          for (mi, c), counts in transitions.items()],
                "reference_policies": reference_outcomes(config)}
    return analysis, episodes, rounds


def reference_outcomes(config):
    """Compute the study's deterministic references without separate run folders."""
    outcomes = []
    for policy in REFERENCE_POLICIES:
        workshop = Workshop(config)
        while not workshop.done:
            actions = None if workshop.state.shutdown_remaining else policy.choose(workshop.state, config)
            workshop.step(actions)
        outcomes.append(summarise(workshop, policy.name).to_dict())
    return outcomes


def write_analysis(run_directory, output_directory):
    output_directory = Path(output_directory)
    analysis, episodes, rounds = analyze(run_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, value in [("analysis.json", analysis), ("episode-data.json", episodes), ("round-data.json", rounds)]:
        (output_directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def number(value, digits=2):
        return "NA" if value is None else f"{value:.{digits}f}"

    lines = ["# Main experiment tables", "", "Workshop results include finished 30-round workshops, even when recognition is incomplete.", "",
             "| Model | Condition | Complete /10 | Workshops /10 | Points mean (SD) | 95% bootstrap interval | Maintenance | Concentration | Breakdowns | Lost rounds | WWR |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cell in analysis["cells"]:
        m = cell["metrics"]; points = m["group_points"]; ci = points["ci95"]
        interval = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "NA"
        name = "Qwen" if cell["model_index"] == 0 else "Phi"
        lines.append(f"| {name} | {cell['condition']} | {cell['complete']} | {cell['workshop_complete']} | "
                     f"{number(points['mean'])} ({number(points['sd'])}) | {interval} | "
                     f"{number(m['maintenance_turns']['mean'])} | {number(m['maintenance_concentration']['mean'], 3)} | "
                     f"{number(m['failure_events']['mean'])} | {number(m['lost_rounds']['mean'])} | "
                     f"{number(m['wasted_wear_reduction']['mean'])} |")
    lines += ["", "| Model | Comparison | Matched pairs | Points difference | 95% bootstrap interval | Maintenance difference | Lost-round difference |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for contrast in analysis["contrasts"]:
        m = contrast["metrics"]; ci = m["group_points"]["ci95"]
        interval = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "NA"
        name = "Qwen" if contrast["model_index"] == 0 else "Phi"
        lines.append(f"| {name} | {contrast['contrast']} | {contrast['n_pairs']} | "
                     f"{number(m['group_points']['mean'])} | {interval} | "
                     f"{number(m['maintenance_turns']['mean'])} | {number(m['lost_rounds']['mean'])} |")
    (output_directory / "computed-tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return analysis
