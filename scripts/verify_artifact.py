#!/usr/bin/env python3
"""Validate the selective artifact using saved values only; never run an LLM."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cogadapt import selection_glm, selection_qwen
from cogadapt.alignment import rank_correlation


def csv_rows(relative):
    with (ROOT / relative).open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(observed, expected, label, atol=1e-9):
    if not math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=atol):
        raise AssertionError(f"{label}: {observed} != {expected}")


def verify_hashes():
    manifest = json.loads((ROOT / "provenance/results_manifest.json").read_text())
    for entry in manifest["exports"]:
        path = ROOT / entry["path"]
        assert path.resolve().is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
    excerpts = json.loads((ROOT / "provenance/code_excerpts.json").read_text())
    for entry in excerpts["excerpts"]:
        text = (ROOT / entry["release_file"]).read_text()
        symbols = {}
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef):
                symbols[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols[target.id] = node
        node = symbols[entry["symbol"]]
        actual = "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno])
        assert hashlib.sha256(actual.encode()).hexdigest() == entry["excerpt_sha256"], entry
    return len(manifest["exports"]), len(excerpts["excerpts"])


def verify_accuracy():
    settings = json.loads((ROOT / "configs/paper_settings.json").read_text())
    runs = csv_rows("results/rq3/paper_runs.csv")
    assert len(runs) == 24 and len({r["run_id"] for r in runs}) == 24
    outcomes = defaultdict(dict)
    for row in csv_rows("results/rq3/task_outcomes.csv"):
        assert row["passed"] in {"0", "1"}
        assert row["task_id"] not in outcomes[row["run_id"]]
        outcomes[row["run_id"]][row["task_id"]] = int(row["passed"])
    assert set(outcomes) == {r["run_id"] for r in runs}
    grouped = defaultdict(list)
    for row in runs:
        values = outcomes[row["run_id"]]
        assert len(values) == int(row["n_tasks"])
        assert sum(values.values()) == int(row["n_correct"])
        close(row["pass_at_1_percent"], 100 * sum(values.values()) / len(values), row["run_id"])
        assert int(row["training_seed"]) == 90317 and int(row["generation_pass"]) == 1
        grouped[row["model"], row["benchmark"], row["condition"]].append(row)
    for row in csv_rows("results/rq3/table2_accuracy.csv"):
        group = grouped[row["model"], row["benchmark"], row["condition"]]
        rates = [float(r["pass_at_1_percent"]) for r in group]
        close(row["pass_at_1_percent"], np.mean(rates), "Table 2 mean")
        if row["condition"] == "Random-K6":
            assert len(group) == 3
            expected_seeds = settings["paper_random_mask_seeds"][f"{row['model']}_{row['benchmark']}"]
            observed = sorted(int(float(r["mask_seed"])) for r in group)
            assert observed == expected_seeds
            close(row["sample_sd_pp"], np.std(rates, ddof=1), "Random sample SD")
        else:
            assert len(group) == 1 and row["sample_sd_pp"] == ""
    for row in csv_rows("results/rq3/paired_dynamic_vs_regular.csv"):
        stem = f"{row['model']}_{row['benchmark']}_"
        d, r = outcomes[stem + "dynamic"], outcomes[stem + "regular_ft"]
        assert d.keys() == r.keys()
        counts = Counter((d[key], r[key]) for key in d)
        for field, pair in [("dynamic_only", (1, 0)), ("regular_only", (0, 1)),
                            ("both_correct", (1, 1)), ("both_wrong", (0, 0))]:
            assert int(row[field]) == counts[pair]
        close(row["delta_pass_pp"], 100 * (counts[1, 0] - counts[0, 1]) / len(d), "Paired delta")
    return len(runs), sum(map(len, outcomes.values()))


def verify_alignment():
    regions = csv_rows("results/rq1/region_values.csv")
    assert len(regions) == 245 and len({r["region_id"] for r in regions}) == 245
    assert len({r["program_id"] for r in regions}) == 32
    table = csv_rows("results/rq1/table1_alignment.csv")
    assert len(table) == 20
    for row in table:
        if row["human_signal"] == "Region attention":
            suffix = {"Expert-choice confidence": "confidence", "MoE write": "moe_write",
                      "Integration": "integration"}[row["model_signal"]]
            field = row["model"].lower() + "_" + suffix
            result = rank_correlation([r["human_attention"] for r in regions], [r[field] for r in regions])
            close(row["value"], result, "Region correlation")
        elif row["human_signal"] == "Regression landing":
            assert row["statistic"] != "Spearman rho"
            assert row["ci_low"] == row["ci_high"] == ""
        else:
            assert float(row["ci_low"]) <= float(row["value"]) <= float(row["ci_high"])
    programs = csv_rows("results/rq2/program_values.csv")
    assert len(programs) == 32 and len({r["program_id"] for r in programs}) == 32
    for row in csv_rows("results/rq2/program_correlations.csv"):
        result = rank_correlation([r[row["x"]] for r in programs], [r[row["y"]] for r in programs])
        close(row["spearman_rho"], result, "Program correlation")
    cells = csv_rows("results/rq2/theta_depth_time_cells.csv")
    assert len(cells) == 5700
    assert len({(r["model"], r["family"], r["block"], r["reading_bin"]) for r in cells}) == 5700
    missing = 0
    for row in cells:
        if row["length_controlled_spearman_rho"] == "":
            missing += 1
            assert row["model"].lower().startswith("glm") and int(row["block"]) == 0
        else:
            assert -1 <= float(row["length_controlled_spearman_rho"]) <= 1
    assert missing == 36
    return len(regions), len(programs), len(cells)


def verify_masks_and_efficiency():
    examples = json.loads((ROOT / "results/rq3/selection_examples.json").read_text())
    assert len(examples) == 6
    for example in examples:
        module = selection_qwen if example["model"] == "qwen" else selection_glm
        metrics = {k: np.asarray(v, dtype=float) for k, v in example["metrics"].items()}
        scores = module.combine_metrics(metrics)
        np.testing.assert_allclose(scores, example["sensitivity_scores"], rtol=1e-10, atol=1e-10)
        actual = module.select_active_blocks(scores, example["difficulty_z"], hard_blocks=10)
        assert actual == example["active_blocks"] and len(actual) == example["width"]
    for row in csv_rows("results/rq3/table3_training_efficiency.csv"):
        close(row["adaptation_reduction_percent"],
              100 * (1 - float(row["eligible_parameters"]) / float(row["installed_parameters"])),
              "Eligibility reduction")
        for field, left, right in [
            ("time_reduction_percent", "dynamic_training_s", "regular_training_s"),
            ("energy_reduction_percent", "dynamic_gpu_wh", "regular_gpu_wh"),
        ]:
            close(row[field], 100 * (1 - float(row[left]) / float(row[right])), field)
    return len(examples)


def main():
    exported, excerpts = verify_hashes()
    runs, outcomes = verify_accuracy()
    regions, programs, cells = verify_alignment()
    masks = verify_masks_and_efficiency()
    print(json.dumps({
        "status": "verified", "hashed_exports": exported, "verbatim_excerpts": excerpts,
        "paper_runs": runs, "binary_outcomes": outcomes, "regions": regions,
        "programs": programs, "heatmap_cells": cells, "real_cached_masks": masks,
        "models_trained_or_evaluated": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
