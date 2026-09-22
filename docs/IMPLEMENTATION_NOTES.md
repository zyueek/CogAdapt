# Implementation and interpretation notes

This is a deliberately partial review release tied to the supplied 21-page
manuscript. It preserves executed definitions rather than silently changing code
to match simplified prose. The full preprocessing, extraction, training, and
evaluation runners are not included.

## Selection

The five block features are standardized across blocks within an example and
combined with weights 0.15 (confidence), 0.25 (expert-choice change), and 0.20 each
for hidden shift, MoE write, and residual integration. Fixed depth bonuses and a
hard-example anchor are present in the extracted selectors.

The score is a hand-weighted frozen-model composite, not a newly measured
EEG–block correlation for each downstream task. GLM's depth preferences are
translated from the Qwen design. A selected transformer block is not an expert.

The width rule uses thresholds −0.5 and +0.5 with widths 4/6/10. Qwen training
uses an EEG-teacher proxy; GLM training and both inference implementations use
model-derived difficulty. This architecture difference is more specific than
the manuscript's general human-difficulty description. Inference difficulty was
standardized across the scored prompt collection; independent online deployment
would need a specified calibration policy.

GLM unavailable routed metrics at dense block 0 are median-imputed for ranking.
In the Figure 5 correlation maps, not-applicable router/MoE cells remain missing
and gray, not zero. The selection and visualization missing-value policies serve
different purposes.

## Human-informed objective

Let z_i be the EEG-teacher proxy and q_i the target-similarity weight:

$$
\widetilde a_i=q_i\operatorname{clip}(e^{0.8z_i},0.4,2.0),
\qquad a_i=\widetilde a_i/\operatorname{mean}_j\widetilde a_j.
$$

For token category salience s_it and hotspot indicator H_it:

$$
\widetilde w_{it}=\operatorname{clip}(e^{0.45s_{it}},0.65,1.5)
\,1.35^{H_{it}},\qquad
w_{it}=\widetilde w_{it}/\operatorname{mean}_{u\in C_i}\widetilde w_{iu}.
$$

With completion positions C_i:

$$
\mathcal L_i=\frac{a_i}{|C_i|}
\sum_{t\in C_i}w_{it}
\left[-\log P_\theta(y_{it}\mid x_i,y_{i,<t})\right].
$$

The loss uses shifted causal logits, ignores prompt labels, and divides by the
supervised-token count. There is no explicit EEG reconstruction loss.

The supplied token_category and hotspot routines show the actual transfer:
token-piece categories plus regex-based source-line heuristics. These functions
do **not** implement cross-language AST-to-AST token matching. Program-level
structural teacher features and token-category transfer are different objects.
The manuscript's AST/syntactic-transfer wording should be interpreted with this
implementation qualification.

Task weights are normalized over unique records before target-matched
oversampling. Final task and token weights need not remain within the
pre-normalization clipping ranges. Regular FT retains target-similarity task
weights and target-matched exposure, but disables human and hotspot weighting.

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
| RQ3 selector hidden_shift | Norm of the same token's block output-minus-input, divided by input norm |
| RQ3 residual_integration | Cosine between the MLP/MoE write and total block update |
| RQ1 region Integration | Mean of region-standardized hidden, MoE-write, and residual components |

Thus Equation 7's illustrative adjacent-token distance is not the RQ3
hidden_shift selector definition. The update/input-state norm described around
Equation 10 is not the selector's residual-integration cosine. Region Integration
is a composite, not one raw activation.

Qwen RQ1/Figure 5 uses multi-metric composites; GLM uses its native scalar
features. Their conceptual roles are comparable, but their formulas are not
identical. Qwen local blocks 31–39 correspond to GLM 30–38 for the region analysis;
the token/event GLM summaries use block 31.

The historical Qwen RQ3 confidence hook applies softmax to a tensor that the
pinned Transformers 5.4.0 router already returned as probabilities. The recorded
selector therefore uses a double-softmax confidence statistic. It is preserved;
silently correcting it would change the experiment. GLM uses normalized sigmoid
scores, with a frozen correction bias for expert selection. Neither should be
treated as a cross-architecture common numerical scale.

## Statistical scope

- Table 1 token and region entries are Spearman correlations. Its regression
  landing entries are mean standardized differences, **not** correlations.
- Token confidence intervals are saved pointwise program-cluster bootstrap
  intervals. No interval is invented for a region or event entry.
- Regions share programs and participants. The 245 rows are not 245 independent
  people or trials. Their correlations are descriptive.
- Figure 5 removes linear code-size effects from original-valued model and theta
  vectors before calculating Spearman correlation on residuals. It does not
  residualize ranks. Every positive, inverse, and unavailable cell is retained.
- Figure 6's block 41 was selected exploratorily across Qwen blocks for that
  metric. Its large association is not a held-out optimal-block confirmation.
- The RQ2 router difficulty is an out-of-fold reading-time prediction, not the
  RQ3 per-example selector composite and not a direct EEG measurement.
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
