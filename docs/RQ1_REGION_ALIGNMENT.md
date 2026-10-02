# Four-category Figure 4: displayed data, generation, and correlation calculation

Figure: [PNG](../figures/region_retention/figure4_retained_four_categories.png) · [PDF](../figures/region_retention/figure4_retained_four_categories.pdf)

This figure shows **81 distinct code regions, numbered 1–81**, grouped into four categories.
Only these displayed regions and their 733 matched token inputs are exported.
IDs follow the existing displayed-region row order; coefficients, categories,
and the figure are unchanged.
The upper panels compare pooled gaze attention with Qwen and GLM integration;
the lower panels compare EEG theta-band power with those same model signals.
Each colored bar is a percentage distribution of **region-level correlations**.
The numeric correlations are calculated across matched tokens **within each
region**, before selecting regions or assigning the displayed categories.

## EEG terminology

The figure uses **EEG theta-band power** as the short panel label. The full
description of the actual feature is:

> Participant-standardized, regression-locked EEG theta-band log-power change
> (4–8 Hz), averaged over 0–1 s relative to a −0.8 to −0.2 s baseline.

Theta oscillations/theta rhythm describe the frequency-band activity. “Theta
wave” is less precise here because the plotted association uses a power-change
feature rather than the raw EEG voltage waveform. The 4–8 Hz theta-band
convention also appears in the [MNE documentation](https://mne.tools/stable/auto_tutorials/epochs/20_visualize_epochs.html).
Band boundaries can differ between studies; 4–8 Hz is the actual band in this
analysis, not a claim of a uniquely mandated range.

The CSV column remains `eeg_theta` and coefficient columns remain
`rho_theta_qwen` / `rho_theta_glm` for compatibility. This is a naming change,
not a change in measurement or numerical results.

## Numerical data for the displayed regions

Displayed-region numerical inputs are committed under [results/rq1](../results/rq1).

| File | Contents |
|---|---|
| [Displayed regions](../results/rq1/figure4_retained_four_categories_regions.csv) | 81 rows: all four coefficients, region/program IDs, final category, token count, previous membership, additions and assignment changes |
| [Displayed token inputs](../results/rq1/figure4_retained_four_categories_token_signals.csv) | 733 rows: every matched token input used in the 81 displayed regions, plus within-region ranks |
| [Exact bar data](../results/rq1/figure4_retained_four_categories_percentages.csv) | 80 rows: 4 panels × 4 categories × 5 bands, with counts, denominators and unrounded percentages |
| [Category denominators](../results/rq1/figure4_retained_four_categories_denominators.csv) | Category sizes, previous/additional region counts and nonpositive counts in each panel |
| [Allowed code predicates](../results/rq1/allowed_code_predicates.csv) | 81 displayed regions only: eligibility for the underlying code-operation labels; scalar/other is a fallback |
| [Correlation provenance](../provenance/rq1_correlation_generation.json) | Component normalization values, original source hashes, token-matching details and upstream analysis description |
| [Selection provenance](../provenance/rq1_region_selection.json) | Category-merging definitions, solver results, source hashes and previous-figure preservation checks |
| [Worked example](../results/rq1/worked_example_region1.csv) | All eight token rows and ranks for region 1 |
| [Reproduction script](../scripts/reproduce_four_category_figure.py) | Standalone verification from the token data and optional redraw from the verified bar data |

The package contains all numerical inputs needed to recompute the displayed correlations
and draw this figure. It does **not** contain raw EEG recordings, raw gaze events,
participant identifiers or source-code text. Rebuilding the preceding EEG/gaze
preprocessing and model composites requires the original workspace caches listed
in the provenance records below.

### Token-input column dictionary

| Column | Meaning |
|---|---|
| `region_id`, `program_id` | Code-region integers 1–81 and pseudonymized program IDs; these are not anatomical brain regions |
| `token_id` | Unique pseudonymized matched-span ID, e.g. `1_T001` |
| `token_order_in_region` | Source order among the common token spans; not the complete tokenizer sequence |
| `n_eeg_participants` | Number of participants contributing theta data to that token |
| `human_attention` | Pooled gaze-attention composite at the token |
| `eeg_theta` | Participant-averaged standardized theta-band log-power-change feature at the token |
| `qwen_integration`, `glm_integration` | Each model's integration composite on that same source span |
| `*_rank` | Ascending average-tie rank of the corresponding signal within this region |
| `semantic_type` | Original six-section label |
| `alternative_section` | Final four-category label |
| `region_estimable` | At least five common tokens and all four finite, nonconstant correlations |
| `included_in_figure` | Whether this region/token contributes to the displayed figure |
| `exclusion_reason` | Empty for every published row, since only displayed regions are exported |

## How the token signals are constructed

### Gaze attention

Use the existing participant-standardized dwell, regression and revisit token
signals. Standardize each component again across all **1,817 common token spans**
using its mean and sample standard deviation (ddof = 1), then take their
equal-weight mean:

```text
attention(token) = [z(dwell) + z(regression) + z(revisit)] / 3
```

The normalization is fixed before any region or sign filtering. It is not
refitted on the favorable 81-region subset. Gaze is pooled across the available
participants, not measured on the participant-validation holdout.

### Model integration

For each model, use hidden movement, MoE write and residual update. Standardize
each of these components over the same full matched-token universe, using sample
SD, and average with equal weights:

```text
integration_model(token) = [z(hidden) + z(MoE write) + z(residual)] / 3
```

Qwen uses the cached block composite (blocks 31–39). GLM averages blocks 30–38,
then maps native GLM token features onto the same Qwen source spans using
character-overlap weights. Qwen and GLM token IDs are not equated. The stored
analysis verified complete GLM coverage for all 1,817 common spans.

### EEG theta-band power change

The original cached EEG extraction performs these operations:

1. Use the preprocessed central/posterior ROI signals: left/mid/right central
   and left/mid/right posterior.
2. Apply a fourth-order Butterworth 4–8 Hz bandpass with forward/backward
   filtering. Compute the squared magnitude of the Hilbert analytic signal,
   then its natural logarithm with a numerical floor of `1e-12`.
3. Take the median log power across the six ROIs at each time point.
4. Time-lock to the start of a regression landing fixation: a fixation reached
   after the eyes move backward in the code. Subtract the mean log power during
   −0.8 to −0.2 s, then average the change over 0–1 s after the landing. The
   original cache applies its recording/epoch quality checks before this step.
5. Z-standardize event values within each participant across their cached
   accepted regression events (population SD, ddof = 0).
6. Average repeated events within participant/token, then average contributing
   participants equally at that token. A participant with many events at a token
   does not receive extra weight in the final between-participant mean.

The resulting scalar is `eeg_theta`. A negative value means below the relevant
standardized reference; it does not mean negative physical power. Token-level
participant coverage varies and can be one participant. This figure does not
use the stricter split-participant coverage filter from the separate validation.

## How each correlation is calculated

For one code region with n common token spans, form two paired vectors. For
example, the EEG–Qwen panel uses:

```text
x = [theta at token 1, ..., theta at token n]
y = [Qwen integration at token 1, ..., Qwen integration at token n]
```

Require **n ≥ 5** and nonconstant, finite signals. Rank each vector in ascending
order, assigning average ranks to tied values. Compute the Pearson correlation
of the two rank vectors; this is **Spearman's rank correlation**:

```text
rho = sum[(rank(x_i) - mean_rank_x) × (rank(y_i) - mean_rank_y)]
      / sqrt(sum[(rank(x_i) - mean_rank_x)^2]
             × sum[(rank(y_i) - mean_rank_y)^2])
```

Repeat on the same token positions for four pairs:

| Panel | First vector | Second vector |
|---|---|---|
| Attention–Qwen | `human_attention` | `qwen_integration` |
| Attention–GLM | `human_attention` | `glm_integration` |
| EEG theta-band power–Qwen | `eeg_theta` | `qwen_integration` |
| EEG theta-band power–GLM | `eeg_theta` | `glm_integration` |

Each region contributes exactly one signed coefficient to each panel. Positive
rho means tokens ranked higher in one signal tend to rank higher in the other;
negative rho means the ordering tends to reverse. This is a within-region,
across-token association, not a correlation across the four category averages.

### Worked example: region 1

The eight rows in [worked_example_region1.csv](../results/rq1/worked_example_region1.csv) give:

| Pair | Spearman rho |
|---|---:|
| Attention–Qwen | 0.1190476190 |
| Attention–GLM | 0.0476190476 |
| EEG theta-band power–Qwen | −0.0238095238 |
| EEG theta-band power–GLM | 0.4285714286 |

Thus region 1 contributes to the gray band in the EEG–Qwen panel and the
0.30–0.50 band in the EEG–GLM panel. The 75% requirement applies to the category's
proportion of positive regions, not to every coefficient of every retained region.

## Selection and category assignment

The published 81 regions were selected using their observed correlation signs
and code-supported category assignments. Regions nonpositive in at least three
panels were excluded upstream. Categories combine branches/loops into **Control
flow**, returns/calls into **Calls / returns**, with **Array expressions** and
**Scalar / other** retained separately. Scalar/other is a fallback.

Each selected region appears once, and the same membership applies to all four
panels. Selection required at least five regions per category and at least 75%
positive coefficients in every category/panel. The optimizer maximized retention
while preserving a preceding selected subset, then minimized assignment changes
and nonpositive region/panel pairs. The saved figure includes 68 previously
selected regions plus 13 additions; three prior category assignments changed.
This history explains the annotations in the unchanged figure. Non-displayed
region rows and alternative category schemes are not exported.

| Category | Total denominator | Previous regions | Additional regions |
|---|---:|---:|---:|
| Control flow | 28 | 19 | 9 |
| Calls / returns | 24 | 21 | 3 |
| Array expressions | 12 | 11 | 1 |
| Scalar / other | 17 | 17 | 0 |
| **Total** | **81** | **68** | **13** |

## How the bars are calculated

Use five bins: `rho ≤ 0`, `0 < rho < 0.30`, `0.30 ≤ rho < 0.50`,
`0.50 ≤ rho < 0.70`, and `0.70 ≤ rho ≤ 1.00`. The first is gray; the positive
bins progress from light to dark blue. Bin on unrounded coefficients.

```text
segment percentage = 100 × number of retained regions in the bin
                           / retained regions in this category
```

For example, 7 of 28 Control-flow regions are nonpositive in each panel, giving
a 25% gray segment and 75% positive total. A bar sums to 100%. The number above
it is the sum of its four positive bins. Segment labels are rounded to one
decimal and labels below 6% are omitted for readability; every exact value is
in the percentage CSV. The y-axis is a proportion of regions, not rho itself.

## Verify and redraw from the exported data

From the repository root, install `requirements.txt`, then run:

```bash
python scripts/reproduce_four_category_figure.py
python scripts/plot_review_figures.py --figure 4
```

The first command recomputes all **324 correlations** for the 81 displayed
regions, checks IDs 1–81 and all 733 token rows and all 80 bar segments. The second
redraws the published PNG/PDF under `figures/region_retention/`.
You can also use `python scripts/reproduce_four_category_figure.py --redraw
--output-dir /tmp/cogadapt-figure4` on a single line to write elsewhere.

The package does not require raw EEG, original source code, or optimization to
reproduce the saved coefficients and display. CSVs should be read with pandas
`float_precision="round_trip"` to preserve bin-boundary values.

## Provenance and upstream scope

[Correlation provenance](../provenance/rq1_correlation_generation.json) records
fixed normalization values and upstream source hashes. Normalization used the
original matched-token universe; it has not been refitted on the selected data.
[Selection provenance](../provenance/rq1_region_selection.json) records the
four-category scheme, solver diagnostics, and new ID convention.
[The release manifest](../provenance/results_manifest.json) contains hashes of
all published data and the figure.

Original source filenames/hashes document upstream processing; those files are
not required by the portable verification command. Rebuilding EEG/gaze
preprocessing requires the original recordings and events, which are excluded.
The reproduction script uses the saved selected set, rather than rerunning
optimization. Region IDs 1–81 are labels and are never inputs to the correlation
calculation. Every token ID uses its new region ID, for example `1_T001`.

## Interpretation

These 81 regions are an **outcome-selected subset**, not a representative sample. The at-least-75%-positive appearance is an explicit
selection constraint. It is not a significance test or new evidence that EEG
alignment improved. Short token sequences can yield unstable correlations; the
separate participant-validation analysis found weak between-group consistency.
The colors encode coefficient ranges, not statistical significance or causation.
