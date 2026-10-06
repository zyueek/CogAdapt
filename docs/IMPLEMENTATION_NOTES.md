# Implementation and interpretation notes

This is a deliberately partial review release tied to the supplied 21-page
manuscript. It preserves executed definitions rather than silently changing code
to match simplified prose. The full preprocessing, extraction, training, and
evaluation runners are not included.

## Gradient masking is different from forward masking

During SFT, _mask_trainable changes requires_grad on attention LoRA and original
router gate parameters. Expert weights remain frozen. Nonselected learned states
are not switched off in that training forward pass.

$$
g_b=\frac1{8}\sum_{i\in\text{accumulation window}}
\mathbf1[b\in M_i]\nabla_{\phi_b}\mathcal L_i.
$$

The optimizer can update the union of example masks across the accumulation
window. This expression describes gradient eligibility, not the complete
stateful AdamW update. Both dynamic and Regular FT install attention adapters
throughout their candidate block pools.

At inference, _activate_blocks sets selected LoRA scaling on, others off, and
restores nonselected routers to base weights. This training/inference asymmetry
is retained in the released code. Neither routine skips transformer blocks.

## Feature names are not interchangeable across analyses

| Context | What the saved quantity means |
|---|---|
| Qwen RQ1 and Figure 5 “hidden-state shift” | A frozen composite of adjacent-token hidden displacement and cosine-distance metrics |
| GLM RQ1/Figure 5 hidden-state shift | Adjacent output-state distance divided by the mean output-state norm |
| Figure 6 Qwen block 41 | One scalar: adjacent-token hidden cosine distance, not the composite |
| RQ1 region Integration | Mean of hidden, MoE-write, and residual components standardized over all 1,817 matched tokens |

Region Integration is a composite, not one raw activation.

Qwen RQ1/Figure 5 uses multi-metric composites; GLM uses its native scalar
features. Their conceptual roles are comparable, but their formulas are not
identical. Qwen local blocks 31–39 correspond to GLM 30–38 for the region analysis;
the replaced historical token/event summaries used block 31.

## Statistical scope

- RQ1 uses within-region Spearman correlations across matched tokens. Its
  81-region figure is selected using observed signs and code-supported category
  assignments; only these displayed regions, numbered 1–81, are exported. The requirement
  of at least 75% positive per bar is a selection constraint, not significance.
- Regions share programs and participants. Pooled token inputs allow numerical
  verification, but do not provide independent participant-level replication.
- See [RQ1 data and methods](RQ1_REGION_ALIGNMENT.md) for the exact selection,
  signal definitions, normalization, denominators, and reproduction commands.
- Figure 5 removes linear code-size effects from original-valued model and theta
  vectors before calculating Spearman correlation on residuals. It does not
  residualize ranks. Every positive, inverse, and unavailable cell is retained.
- Figure 6's block 41 was selected exploratorily across Qwen blocks for that
  metric. Its large association is not a held-out optimal-block confirmation.
- The RQ2 router difficulty is an out-of-fold reading-time prediction, not a
  direct EEG measurement.
- Table 2 uses exactly the paper's three Random-K6 masks per setting. These are
  listed in the settings and per-run tables; this is not a full search archive.
- Table 3's reduction is per-example gradient eligibility, not model-size or
  backbone-FLOP reduction. Training costs come from instrumented replicas and
  exclude the offline human-data and sensitivity stages.
- The saved paired q-values retain the original 20-comparison family. They are
  not recomputed as a four-comparison family for this release.
- Single-seed training and historical configuration selection limit causal and
  confirmatory claims. The BigCodeBench dynamic-versus-FT confidence intervals
  include zero. No EEG or gaze is an inference input.
