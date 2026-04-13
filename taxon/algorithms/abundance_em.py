"""Taxon abundance estimation via multinomial-mixture EM (plugin wrapper).

Wraps :mod:`taxon.algorithms.abundance_em_core` so it can be discovered by
the ProteomicsAgent ``TaxonRegistry``. The plugin estimates a continuous
relative-abundance vector on the probability simplex; unlike presence-only
methods, it produces a quantitative breakdown of how much each taxon
contributed to the observed peptide evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path

from taxon.base_plugin import TaxonPlugin, TaxonResult

from .abundance_em_core.detectability import DetectabilityWeights
from .abundance_em_core.identifiability import identifiability_report
from .abundance_em_core.mapping_matrix import build_mapping_matrix
from .abundance_em_core.model import AbundanceEM

logger = logging.getLogger(__name__)


class AbundanceEMPlugin(TaxonPlugin):
    """Estimate relative taxon abundances by EM on a multinomial mixture.

    Required config keys
    --------------------
    fasta_path : str
        Path to the protein FASTA database used during the search.

    Optional config keys
    --------------------
    pepxml_path : str
        Path to a validated pepXML file.  When provided the plugin extracts
        peptide-level spectral counts and protein mappings directly from the
        pepXML, overriding both the *peptides* argument and any
        ``spectral_counts`` dict in the config.
    exclude_prefixes : list[str]
        Protein-accession prefixes to exclude during both pepXML parsing and
        FASTA parsing (default ``["DECOY", "contag"]``).
    alpha : float
        Dirichlet prior hyperparameter (default ``0.5``, sparsity-inducing).
    max_iter : int
        Maximum EM iterations (default ``500``).
    tol : float
        L1 convergence tolerance on ``pi`` updates (default ``1e-6``).
    n_restarts : int
        Number of random EM restarts (default ``1``).
    min_abundance : float
        Post-fit threshold; taxa below this are zeroed and the vector is
        renormalized (default ``1e-4``).
    enzyme : str
        Digestion enzyme (default ``"trypsin"``).
    missed_cleavages : int
        Allowed missed cleavages (default ``2``).
    min_peptide_length, max_peptide_length : int
        Length filters for in-silico peptides (defaults ``7`` / ``50``).
    run_identifiability : bool
        Whether to run the identifiability report and log warnings
        (default ``True``).
    spectral_counts : dict[str, int]
        Optional mapping ``peptide_sequence -> count``. Ignored when
        ``pepxml_path`` is set. If absent, every observed peptide is given
        a count of 1.
    seed : int
        RNG seed for reproducibility (default unset).
    pure_culture_pepxml_paths : dict[str, str]
        Mapping ``culture_label -> pepxml_path`` for pure culture experiments.
        When provided, per-peptide detectability weights are estimated from
        the pure culture spectral counts and used to correct the emission
        matrix.  This is the recommended way to improve accuracy when pure
        culture data is available alongside the mixed sample.
    detectability_weights : dict[str, float]
        Direct specification of per-peptide detectability weights as a
        mapping ``peptide_sequence -> weight``.  Overrides
        ``pure_culture_pepxml_paths`` if both are provided.
    detectability_pseudocount : float
        Pseudocount for pure culture weight estimation (default ``0.5``).
    """

    name = "abundance_em"
    description = (
        "Estimates relative taxon abundance from peptide spectral counts "
        "using a multinomial mixture model with EM inference and sparse "
        "Dirichlet prior."
    )
    requires_internet = False

    # ----------------------------------------------------------------- API

    def validate_config(self, config: dict) -> bool:
        """Check that ``fasta_path`` (and optionally ``pepxml_path``) exist."""
        fasta_path = config.get("fasta_path")
        if not fasta_path:
            return False
        if not Path(str(fasta_path)).is_file():
            return False
        pepxml_path = config.get("pepxml_path")
        if pepxml_path and not Path(str(pepxml_path)).is_file():
            return False
        return True

    def run(self, peptides: list, config: dict) -> list:
        """Run abundance estimation and return TaxonResult records.

        Returns
        -------
        list of TaxonResult
            Sorted by abundance descending. Only taxa with abundance strictly
            greater than ``min_abundance`` are included.
        """
        fasta_path = str(config["fasta_path"])
        pepxml_path = config.get("pepxml_path")
        exclude_prefixes = config.get("exclude_prefixes", ["DECOY", "contag"])

        # Plugin parameters with defaults.
        alpha = float(config.get("alpha", 0.5))
        max_iter = int(config.get("max_iter", 500))
        tol = float(config.get("tol", 1e-6))
        n_restarts = int(config.get("n_restarts", 1))
        min_abundance = float(config.get("min_abundance", 1e-4))
        enzyme = str(config.get("enzyme", "trypsin"))
        missed_cleavages = int(config.get("missed_cleavages", 2))
        min_length = int(config.get("min_peptide_length", 7))
        max_length = int(config.get("max_peptide_length", 50))
        run_id_check = bool(config.get("run_identifiability", True))
        seed = config.get("seed")

        # When a pepXML is available, derive peptides and spectral counts
        # from the PSM-level data instead of the caller-supplied protein list.
        pepxml_protein_map = None
        if pepxml_path:
            from .abundance_em_core.pepxml_parser import parse_pepxml

            spectral_counts, pepxml_protein_map = parse_pepxml(
                pepxml_path, exclude_prefixes=exclude_prefixes,
            )
            peptides = list(spectral_counts.keys())
        else:
            spectral_counts = config.get("spectral_counts") or {}

        if not peptides:
            logger.warning("AbundanceEMPlugin: empty peptide list, returning []")
            return []

        # Build the mapping matrix.
        A, peptide_list, taxon_labels = build_mapping_matrix(
            peptides=peptides,
            fasta_path=fasta_path,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            exclude_prefixes=exclude_prefixes,
            pepxml_protein_map=pepxml_protein_map,
        )
        if A.shape[1] == 0:
            logger.warning(
                "AbundanceEMPlugin: no peptides matched any taxon in %s",
                fasta_path,
            )
            return []

        # Build the spectral count vector aligned with peptide_list.
        y = self._build_count_vector(peptide_list, spectral_counts)

        # Identifiability report (best-effort, non-fatal).
        if run_id_check:
            try:
                report = identifiability_report(
                    A, taxon_names=[lbl.split("|", 1)[1] for lbl in taxon_labels]
                )
                if report["warnings"]:
                    for w in report["warnings"]:
                        logger.warning("identifiability: %s", w)
            except Exception as exc:  # noqa: BLE001
                logger.warning("identifiability report failed: %s", exc)

        # Build detectability weights if pure culture data is available.
        detect_weights_vec = None
        raw_detect_weights = config.get("detectability_weights")
        pure_culture_paths = config.get("pure_culture_pepxml_paths")
        detect_pseudocount = float(config.get("detectability_pseudocount", 0.5))

        if raw_detect_weights:
            # Direct weight specification takes precedence.
            dw = DetectabilityWeights(
                weights=raw_detect_weights,
                default_weight=1.0,
                source="user_provided",
            )
            detect_weights_vec = dw.get_weight_vector(peptide_list)
            logger.info(
                "Using user-provided detectability weights (%d peptides)",
                dw.n_peptides_estimated,
            )
        elif pure_culture_paths:
            try:
                dw = DetectabilityWeights.from_pure_culture_pepxml(
                    pepxml_paths=pure_culture_paths,
                    exclude_prefixes=exclude_prefixes,
                    pseudocount=detect_pseudocount,
                )
                detect_weights_vec = dw.get_weight_vector(peptide_list)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to estimate detectability weights from pure "
                    "cultures: %s. Falling back to uniform weights.",
                    exc,
                )

        # Fit the EM model.
        model = AbundanceEM(
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            n_restarts=n_restarts,
            min_abundance=min_abundance,
            seed=seed,
        )
        model.fit(A, y, detectability_weights=detect_weights_vec)
        logger.info(
            "AbundanceEM converged=%s in %d iterations (final lp=%.4f)",
            model.converged_,
            model.n_iter_,
            model.log_posterior_history_[-1] if model.log_posterior_history_ else float("nan"),
        )

        # Convert to TaxonResult objects.
        results: list = []
        confidences = self._confidences(model.standard_errors_)
        responsibilities = model.responsibilities_

        for col, label in enumerate(taxon_labels):
            abundance = float(model.pi_[col])
            if abundance <= min_abundance:
                continue

            taxon_id, taxon_name = label.split("|", 1) if "|" in label else ("0", label)

            assigned_idx = [i for i in range(A.shape[0]) if responsibilities[i, col] > 0.5]
            assigned_peptides = [peptide_list[i] for i in assigned_idx]

            results.append(
                TaxonResult(
                    taxon_id=taxon_id,
                    taxon_name=taxon_name,
                    rank="species",
                    abundance=abundance,
                    confidence=float(confidences[col]),
                    peptide_count=len(assigned_peptides),
                    peptides=assigned_peptides,
                )
            )

        results.sort(key=lambda r: r.abundance, reverse=True)
        return results

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _build_count_vector(peptide_list: list, spectral_counts: dict):
        """Project a peptide -> count map onto the row order of the matrix."""
        import numpy as np

        if not spectral_counts:
            return np.ones(len(peptide_list), dtype=np.float64)
        y = np.zeros(len(peptide_list), dtype=np.float64)
        for i, pep in enumerate(peptide_list):
            y[i] = float(spectral_counts.get(pep, 0))
        if y.sum() == 0:
            logger.warning(
                "All spectral counts are zero; falling back to count=1 per peptide"
            )
            y = np.ones(len(peptide_list), dtype=np.float64)
        return y

    @staticmethod
    def _confidences(standard_errors):
        """Apply the SE -> confidence heuristic used by the core model."""
        import numpy as np

        if standard_errors is None:
            return np.zeros(0)
        return np.clip(1.0 - 2.0 * standard_errors, 0.0, 1.0)
