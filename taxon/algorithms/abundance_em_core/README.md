# abundance_em_core

Numerical core for the `abundance_em` taxon plugin: builds the
peptide × taxon mapping matrix, runs the multinomial-mixture EM, and
applies post-EM corrections (marker-based cell equivalents and the
proteome-mass correction documented below).

## Modules

| Module | Purpose |
|---|---|
| `mapping_matrix.py` | FASTA / pepXML parsing → peptide × taxon assignment matrix `A`. |
| `model.py` | EM inference for the multinomial-mixture taxon model. |
| `marker_correction.py` | bac120 / ar53 marker-based cell-equivalent correction. |
| `proteome_mass_correction.py` | π → b_t conversion via `b_t ∝ π_t / W_t^α`. |
| `pepxml_parser.py` | Pull spectral counts + protein mappings out of a validated pepXML. |
| `identifiability.py` | Pre-EM warning report on degenerate matrix structure. |
| `detectability.py` | Per-peptide detectability weighting (`uniform` / `sequence_features` / `file`). |
| `matrix_visualization.py` | Diagnostic heatmaps + TSV exports of the A / M / W matrices. |
| `visualize_results.py` | Bar-chart renderer for the unified abundance TSV. |
| `scripts/alpha_ablation.py` | Sweep α over an existing `abundance_results.tsv`. |
| `tests/` | pytest suite. |

## α parameter (genome-scaling exponent)

`compute_biomass_abundance` now exposes a configurable scaling exponent
`α` in the proteome-mass correction:

```
b_t = (π_t / W_t^α) / Σ_{t'} (π_{t'} / W_{t'}^α)
```

| α | Behaviour | Source |
|---|---|---|
| `0` | Disables correction; b_t collapses to π_t after renormalization. | — |
| `1` | Original linear form (`b_t ∝ π_t / W_t`). Bit-identical to the pre-α implementation thanks to a short-circuit. | Pible 2020 *Microbiome*; Kleiner 2017 *Nat. Commun.* (TPA baseline). |
| `4.8` *(default)* | Bacterial genome-volume scaling. Cell volume scales with genome size at log-log slope 0.21 (Kempes 2016, *ISME J*) → cell volume ∝ G⁴·⁸. Per-cell protein concentration `c_p` is a near-universal bacterial constant (Milo 2013, *BioEssays* 35:1050), so per-cell protein biomass ∝ G⁴·⁸. | Kempes 2016, Milo 2013. |

Negative α raises `ValueError`. The α=1 path explicitly uses the
original division (no `np.power` / log-space exp) so a regression test
against the pre-α implementation passes exactly. The α≠1 path uses
log-space exponentiation (`exp(-α · log W)`) to stay numerically stable
for large W and α.

### Wiring through the pipeline

| Layer | Surface |
|---|---|
| Plugin config | `genome_scaling_exponent: float = 4.8` (only used when `proteome_mass_correction: true`). |
| CLI flag | `--genome-scaling-exponent <float>` on `python main.py run` and `run-pipeline`. |
| YAML | `corrections.proteome_mass.genome_scaling_exponent: 4.8`. |
| Env var | `TAXON_GENOME_SCALING_EXPONENT`. |
| Reproducibility | Materialised into the saved `run_config.yaml` whenever proteome-mass correction is on. |
| Diagnostics | One-line `Proteome-mass correction: alpha=…. W_t range: …. b_t / pi_hat ratio range: ….` appended to `diagnostics/biomass_diagnostics.txt`. |

### Ablation script

```
python taxon/algorithms/abundance_em_core/scripts/alpha_ablation.py \
    --input <run>/abundance_results.tsv \
    --alphas 0.0,1.0,2.0,3.6,4.8 \
    --output alpha_ablation/
```

The script reuses the production `compute_biomass_abundance`, so the
formula is not duplicated. Outputs:

* `alpha_ablation_table.tsv` — `taxon, pi_hat, W_t, b_t_alpha_<value>` per α.
* `alpha_ablation_plot.png` — grouped bar chart of top-15 taxa by π_hat.
* `alpha_ablation_summary.txt` — L1 distance and Bray–Curtis dissimilarity
  vs π_hat for each α.
