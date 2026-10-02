# Paper-to-artifact map

The provenance manifests identify the exact local source file hashes. Source
filenames are retained for traceability, but those full source trees are not
included or required to inspect this release.

| Paper location | Included object | Reproducible from this subset |
|---|---|---|
| Sections 4.3–4.5 | Selector, weight, loss, and masking excerpts | Cached-mask selection; numerical weight/loss behavior; masking unit tests |
| RQ1, Figure 4 | Selected 81-region PNG/PDF, 245-region audit, 1,817 pooled token inputs, memberships and bar counts | Recompute all 732 coefficients and redraw the four-category percentage bars; see [instructions](RQ1_REGION_ALIGNMENT.md) |
| Figure 5 | All 5,700 model/family/block/bin cells, including 36 unavailable GLM cells | Redraw the full maps; upstream metric/theta extraction is excluded |
| Figure 6 | 32 pseudonymized program aggregates | Recompute the three rank correlations and redraw scatter panels |
| Introduction's program difficulty–theta result | Saved Qwen/GLM out-of-fold difficulty aggregates | Recompute 0.717 and 0.511 correlations; not refit the predictors |
| Table 2 | 24 paper runs; binary outcomes; three random masks per setting | Recompute pass@1 and random means/sample SDs; verify paired wins/losses |
| Table 3 | Four paired training-resource summaries | Recompute eligibility, time, and energy reductions; no new training |
| RQ3 interpretation | Four saved paired contrast rows | Check wins/losses from outcomes and inspect saved uncertainty |

## Deliberately excluded

- Original study EEG/gaze recordings and participant-level derivatives.
- Full stimuli code, token-level participant histories, and raw model activations.
- The EEG-teacher fitting/preprocessing pipeline and GPU feature extraction.
- Training datasets, benchmark task text, canonical/generated solutions, and
  hidden execution tests.
- LLM weights, LoRA checkpoints, router checkpoints, and model-loading code.
- End-to-end training, inference, evaluation, scheduling, and deployment runners.
- Experiments and random masks outside the paper's explicitly listed cohorts.
- The manuscript PDF, original schematic illustrations, and workspace Git history.

Program IDs P01–P32 derive from source condition indices, not participant
identities. Region IDs R001–R245 preserve the fixed Figure 4 ordering. Benchmark
task IDs preserve pairing across the included conditions but do not distribute
task names or content.

No model performance is claimed from running the supplied unit tests. The
verification/plotting helpers perform inexpensive arithmetic on previously saved
results. The excerpt manifest distinguishes verbatim research functions from
new standalone wrappers and review tooling.
