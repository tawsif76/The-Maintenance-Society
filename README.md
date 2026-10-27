# The Maintenance Society: Coordination of Preventive Work in Small Language Model Agents

[Project overview and author portfolio](https://tawsif76.github.io/projects/the-maintenance-society/) · [Mohammed Sydul Hasan Tawsif](https://tawsif76.github.io/)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Official repository for **"The Maintenance Society: Coordination of Preventive Work in Small Language Model Agents"**.

This benchmark studies how small language model agents allocate effort between private production and shared infrastructure maintenance under repeated operational pressure.

---

## Table of Contents

- [Overview](#overview)
- [Study Design](#study-design)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Reproducing Paper Results](#reproducing-paper-results)
  - [1. Recompute Paper Tables and Metrics](#1-recompute-paper-tables-and-metrics)
  - [2. Inspect Experiment Status and Plans](#2-inspect-experiment-status-and-plans)
  - [3. Run Live Experiments with Ollama](#3-run-live-experiments-with-ollama)
- [Data and Checkpoints Format](#data-and-checkpoints-format)

---

## Overview

When autonomous agents share infrastructure or machinery, keeping equipment operational requires time and effort. An agent that produces an item earns an immediate private reward but increases machine wear. An agent that maintains the machine gives up that reward to reduce wear, protecting future production opportunities for the entire group.

This project investigates three questions:

1. Does showing cumulative maintenance totals on the dashboard change maintenance distribution and group output?
2. Does discussion before final decisions help agents coordinate maintenance timing and prevent redundant effort?
3. Do agents give recognition credit to peers who sacrifice production points for machine upkeep?

---

## Study Design

Each episode contains six agents and lasts 30 rounds. All six agents use the same model configuration, with separate observations and decisions. During an operational round, each agent chooses between two actions:

- **Produce:** earn three personal points and add one unit of machine wear.
- **Maintain:** earn no points and remove up to six units of wear.

The machine starts at wear 12. If wear reaches 30 or above after the simultaneous actions, the machine breaks down. Production points from that round still count. The next two rounds are closed, after which automatic repair resets wear to 12. Closed rounds count toward the episode's 30-round limit.

The four conditions vary the maintenance summary and the opportunity for discussion:

| Condition | Dashboard totals | Discussion before final decisions |
| --- | --- | --- |
| A | Production points | No |
| B | Production points and maintenance counts | No |
| C | Production points | Yes |
| D | Production points and maintenance counts | Yes |

All conditions provide the full completed action history. Agents in A and C can count past maintenance actions themselves; B and D provide that count on the dashboard. The summary adds no maintenance payment or mandatory assignment.

In C and D, each agent submits an intended action and a message of at most 60 whitespace-separated words. All proposals are revealed together. Agents then make private final decisions and may revise their intentions. At the end of the workshop, each agent allocates 100 recognition points among its five peers. Recognition does not change production rewards.

---

## Key Findings

Across 80 scheduled episodes (2 models × 4 conditions × 10 episodes) using local open-weight models (`Qwen3-4B-Instruct-2507` and `Phi-4-mini-instruct`):

- **Dashboard summaries helped Phi without discussion:** Mean output increased from 336.0 in A to 394.5 points in B. Mean maintenance actions per workshop fell from 68.0 to 48.5, with no breakdowns in either condition. In the same A-to-B comparison, Qwen performed less maintenance but experienced more breakdowns and gained no production benefit.
- **Discussion did not provide a consistent coordination benefit:** Finished Phi discussion workshops wasted over 84% of their supplied maintenance capacity and produced less than the no discussion groups. Qwen showed no clear production benefit. 
- **Phi added most maintenance after proposals:** In finished Phi discussion workshops, over 80% of final maintenance actions followed an initial intention to produce. These switches locate much of the maintenance at the final-decision stage, after proposals were revealed. 
- **Equal work did not ensure efficient work:** Finished Phi discussion groups shared maintenance more evenly, but their mean output fell below the always produce reference policy, which performs no maintenance and accepts repeated shutdowns.
- **Recognition showed no consistent relationship with maintenance effort:** The mean maintenance recognition rank correlations varied across models and conditions. Recognition analysis remains exploratory, with especially few complete Phi discussion episodes.

Workshop results use 69 finished workshops. Only four Phi C workshops and five Phi D workshops finished, which limits those comparisons. Recognition results use 50 complete allocation sets.

---

## Repository Structure

```text
maintenance-society/
├── configs/
│   ├── experiment.json        # Workshop parameters, conditions, and schedules
│   └── models.json            # Exact model tags, digests, and generation settings
├── prompts/
│   ├── system.txt             # System prompt containing workshop rules and objective
│   ├── proposal.txt           # Deliberation prompt (message + intended action schema)
│   ├── decision.txt           # Action choice prompt (final decision schema)
│   └── recognition.txt        # End-of-episode peer recognition prompt
├── src/
│   └── maintenance_society/
│       ├── agents/            # Inference client, schemas, observations, controllers
│       ├── environment/       # Deterministic workshop simulator and reference policies
│       ├── evaluation/        # Metrics, aggregation, bootstrap statistics, and tables
│       └── experiments/       # Main runner, scheduler, journals, and storage
├── scripts/
│   ├── analyze_results.py     # Aggregates saved runs and writes paper tables
│   └── run_experiment.py      # Experiment CLI (planning, status inspection, execution)
├── data/
│   └── checkpoints/main/      # Saved main-study records; not included in a fresh clone
├── results/
│   ├── preprint/              # Research manuscript and supplementary results
│   └── tables/main-analysis/  # Computed JSON and Markdown summary tables
├── README.md                  # Project guide
├── pyproject.toml             # Project package configuration and dependencies
└── requirements.txt           # Pinned runtime dependencies
```

---

## Setup and Installation

### Prerequisites

- Python 3.11 or 3.12. The recorded study used Python 3.12.
- [Ollama](https://ollama.com/) **exactly version 0.40.3**, only for live inference. The runner rejects other versions under the current configuration.

Analysis of saved records requires neither Ollama nor a GPU.

### 1. Prepare a virtual environment

Download or clone this repository, then open a terminal in the `maintenance-society` directory. Create the environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For core execution and analysis on Linux/macOS, activate it with:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Core execution and analysis
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

## Reproducing Paper Results

**Study-file availability:** The local study records live under `data/checkpoints/main/`. Git excludes the saved records and computed analysis outputs, so a fresh clone does not contain them. The compiled preprint is included at `results/preprint/latex/main-primearxiv.pdf`; the remaining LaTeX folder stays local. This README does not yet provide a public download link for the saved records.

Before reproducing the analysis, copy the saved main-study records into this location:

```text
data/checkpoints/main/
```

The main run folder must contain its manifest and all 80 episode records, including incomplete attempts. With these files in place, the commands below recreate the analysis outputs without model calls.

### 1. Recompute Paper Tables and Metrics

To aggregate episode records, run paired bootstrap comparisons (20,000 resamples), compute waste fractions, and refresh markdown tables:

```bash
python scripts/analyze_results.py
```

Outputs are written to `results/tables/main-analysis/`:

- `analysis.json`: Full nested metrics, bootstrap confidence intervals, and reference policy outcomes.
- `computed-tables.md`: Aggregate workshop results and paired condition comparisons. It is not a copy of every manuscript table.
- `episode-data.json` and `round-data.json`: Per-episode and per-round records for analysis.

### 2. Run Live Experiments with Ollama

The current runner executes the fixed 80-attempt design using the two configured models. Selecting different models requires code and configuration changes; it is not a command-line option.

To repeat the study with new model calls, use a new output directory so the saved main experiment remains intact:

1. **Pull the target models:**
   ```bash
   ollama pull qwen3:4b-instruct-2507-q4_K_M
   ollama pull phi4-mini:3.8b-q4_K_M
   ```

2. **Execute the experiment runner:**
   ```bash
   python scripts/run_experiment.py --run --output-dir data/checkpoints/custom-run
   ```

Before episode inference, the runner validates the study configuration and checks the Ollama version, installed model digests, and model capabilities. During execution, it validates response formats and reported context usage, and saves progress after each committed round. It does not run a separate preflight experiment or context-probe suite.

An invalid response receives one correction request. A second invalid response ends that attempt as incomplete; the runner does not substitute an action. Repeating the run command resumes unfinished work when the code and configuration match. Terminal attempts, whether complete or incomplete, are not rerun. If all attempts have ended, `--run` reports their status without making model calls.

Analysis defaults to the saved main experiment. To analyse a separate repeat without overwriting its main-study tables, specify both directories:

```bash
python scripts/analyze_results.py --run-dir data/checkpoints/custom-run --output-dir results/tables/custom-run
```

---

## Data and Checkpoints Format

Each episode under `data/checkpoints/main/episodes/<episode_id>/` contains:

- `episode.json`: Immutable episode specification, including the seed, model tag, condition, agent display order, and workshop rules.
- `checkpoint.json`: Latest saved simulation state, completed-round count, recognition progress, and status. Intermediate statuses include `running`, `recognition`, and `interrupted`.
- `result.json`: Terminal outcome, completion status, request statistics, and available metrics. An attempt that stops before round 30 has no full-workshop metrics. A finished workshop can still have incomplete recognition.
- `rounds/{001..030}.json`: Committed round records containing actions, wear before and after the round, breakdowns, and proposals where discussion occurs. Early-stopped attempts contain fewer than 30 records. Closed rounds contain no agent actions.
- `requests/*.json`: Saved requests, responses, validation decisions, and any recorded request failures from the main attempt.
