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
    detectability_mode : str
        Peptide detectability weighting mode (default ``"uniform"``).
        Options: ``"uniform"`` (legacy, no weighting),
        ``"sequence_features"`` (physicochemical prediction),
        ``"file"`` (load from TSV).
    detectability_file : str
        Path to a TSV with pre-computed detectability scores.
        Required when ``detectability_mode="file"``.
    resolve_uniprot : bool
        When ``True`` (default) the FASTA is pre-scanned to resolve bare
        UniProt accession headers (e.g. ``>Q2FYC6``) to their source
        organism via a cached UniProt REST lookup, preventing those
        proteins from being bucketed as ``unclassified``.
    prefix_map_file : str
        Path to a two-column TSV (no header) mapping accession prefixes
        to organism names (``<prefix>\\t<organism>``).  Used to rescue
        FASTA entries whose headers lack organism annotations.
    taxon_level : str
        ``"species"`` (default) or ``"strain"``.  Controls organism-name
        normalisation depth: species mode collapses strain-level variants
        into binomials; strain mode preserves sub-species identifiers.
    output_dir : str
        Directory where auxiliary output files (e.g.
        ``unclassified_entries.tsv``) are written.  When unset, no
        auxiliary files are produced.
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
        # cRAP common-contaminant entries must never enter the EM — they
        # otherwise form a "CRAP" taxon that competes with real community
        # members for PSM mass (header-parsing review, CRITICAL).
        exclude_prefixes = config.get(
            "exclude_prefixes", ["DECOY", "contag", "CRAP_", "cRAP_", "crap_"]
        )

        # Plugin parameters with defaults.
        alpha = float(config.get("alpha", 0.5))
        max_iter = int(config.get("max_iter", 500))
        tol = float(config.get("tol", 1e-6))
        n_restarts = int(config.get("n_restarts", 1))
        # `abundance_threshold` is the YAML / config-file alias for
        # `min_abundance`; both are accepted for backwards compatibility.
        min_abundance = float(
            config.get("min_abundance",
                       config.get("abundance_threshold", 1e-4))
        )
        init_strategy = str(config.get("init_strategy", "unique"))
        enzyme = str(config.get("enzyme", "trypsin"))
        missed_cleavages = int(config.get("missed_cleavages", 2))
        min_length = int(config.get("min_peptide_length", 7))
        max_length = int(config.get("max_peptide_length", 50))
        run_id_check = bool(config.get("run_identifiability", True))
        seed = config.get("seed")
        detectability_mode = str(config.get("detectability_mode", "uniform"))
        detectability_file = config.get("detectability_file")
        resolve_uniprot = bool(config.get("resolve_uniprot", True))
        prefix_map_file = config.get("prefix_map_file")
        output_dir = config.get("output_dir")
        taxon_level = str(config.get("taxon_level", "species"))
        min_psm_threshold = int(config.get("min_psm_threshold", 0))
        generate_plot = bool(config.get("generate_plot", True))
        plot_top_n = int(config.get("plot_top_n", 15))
        unified_table = bool(config.get("unified_table", True))

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
        mapping_result = build_mapping_matrix(
            peptides=peptides,
            fasta_path=fasta_path,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            exclude_prefixes=exclude_prefixes,
            pepxml_protein_map=pepxml_protein_map,
            resolve_uniprot=resolve_uniprot,
            prefix_map_file=str(prefix_map_file) if prefix_map_file else None,
            taxon_level=taxon_level,
        )
        A = mapping_result.matrix
        peptide_list = mapping_result.peptide_list
        taxon_labels = mapping_result.taxon_labels
        unclassified_peptides = mapping_result.unclassified_peptides

        # Write unclassified entries to a diagnostic TSV.
        if unclassified_peptides and output_dir:
            diag_dir = Path(str(output_dir)) / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            tsv_path = diag_dir / "unclassified_entries.tsv"
            with tsv_path.open("w", encoding="utf-8") as fh:
                fh.write("peptide_sequence\tprotein_accession\n")
                for pep, acc in unclassified_peptides:
                    fh.write(f"{pep}\t{acc}\n")
            n_unique = len({pep for pep, _acc in unclassified_peptides})
            logger.info(
                "%d peptide(s) mapped only to unclassified proteins "
                "(excluded from EM). Details written to %s",
                n_unique, tsv_path,
            )

        if A.shape[1] == 0:
            logger.warning(
                "AbundanceEMPlugin: no peptides matched any taxon in %s",
                fasta_path,
            )
            return []

        # Build the spectral count vector aligned with peptide_list.
        y = self._build_count_vector(peptide_list, spectral_counts)

        # ----------------------------------------------------------------- #
        # PRE-EM DIAGNOSTIC BLOCK — remove when no longer needed            #
        # ----------------------------------------------------------------- #
        import numpy as np

        _row_sums = A.sum(axis=1)  # number of taxa each peptide maps to
        _y_total = float(y.sum())

        logger.info("=== PRE-EM DIAGNOSTIC ===")
        logger.info(
            "Total PSMs: %d, Total peptides: %d, Total taxa: %d",
            int(_y_total), len(peptide_list), A.shape[1],
        )

        # Per-taxon stats
        _taxon_rows = []
        for _col, _lbl in enumerate(taxon_labels):
            _name = _lbl.split("|", 1)[-1]
            _col_vec = np.asarray(A[:, _col]).ravel()
            _maps_here = _col_vec > 0
            _unique_mask = _maps_here & (_row_sums == 1)
            _n_mapped = int(_maps_here.sum())
            _n_unique = int(_unique_mask.sum())
            _total_sc = float(y[_maps_here].sum())
            _unique_sc = float(y[_unique_mask].sum())
            _sc_frac = 100.0 * _total_sc / _y_total if _y_total > 0 else 0.0
            _taxon_rows.append((_name, _n_mapped, _n_unique, _total_sc, _unique_sc, _sc_frac))

        _taxon_rows.sort(key=lambda r: r[3], reverse=True)

        logger.info(
            "%-40s %8s %8s %10s %10s %8s",
            "Taxon", "Mapped", "Unique", "TotalSC", "UniqueSC", "SC%",
        )
        for _row in _taxon_rows[:20]:
            _name, _n_mapped, _n_unique, _total_sc, _unique_sc, _sc_frac = _row
            logger.info(
                "%-40s %8d %8d %10d %10d %7.2f%%",
                _name[:40], _n_mapped, _n_unique, int(_total_sc), int(_unique_sc), _sc_frac,
            )

        # Summary stats
        logger.info(
            "Peptides with zero counts: %d | map to 1 taxon: %d | map to >1 taxon: %d",
            int((y == 0).sum()),
            int((_row_sums == 1).sum()),
            int((_row_sums > 1).sum()),
        )
        logger.info(
            "Taxa with zero unique peptides: %d / %d",
            sum(1 for r in _taxon_rows if r[2] == 0), len(_taxon_rows),
        )
        logger.info("=== END PRE-EM DIAGNOSTIC ===")
        # ----------------------------------------------------------------- #

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

        # Fit the EM model.
        model = AbundanceEM(
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            n_restarts=n_restarts,
            min_abundance=min_abundance,
            init=init_strategy,
            seed=seed,
            detectability_mode=detectability_mode,
            detectability_file=detectability_file,
        )
        model.fit(A, y, peptide_sequences=peptide_list)
        logger.info(
            "AbundanceEM converged=%s in %d iterations (final lp=%.4f)",
            model.converged_,
            model.n_iter_,
            model.log_posterior_history_[-1] if model.log_posterior_history_ else float("nan"),
        )

        # --- Marker-based cell-equivalent correction (Cycle 5, opt-in) ---
        marker_result = None
        if config.get("marker_correction", False):
            marker_result = self._run_marker_correction(
                model=model,
                mapping_result=mapping_result,
                spectral_counts=spectral_counts,
                config=config,
            )

        # --- Proteome-mass correction (Cycle 5b, opt-in) ---
        biomass_result = None
        if config.get("proteome_mass_correction", False):
            biomass_result = self._run_proteome_mass_correction(
                model=model,
                mapping_result=mapping_result,
            )

        # Convert to TaxonResult objects.
        results: list = []
        confidences = self._confidences(model.standard_errors_)
        responsibilities = model.responsibilities_

        # Per-taxon total PSM count (sum of y over peptides this taxon can
        # produce, with non-trivial responsibility). Used both for the
        # `min_psm_threshold` filter and for the unified output table.
        taxon_psm_count = self._taxon_psm_counts(A, y, responsibilities)

        for col, label in enumerate(taxon_labels):
            abundance = float(model.pi_[col])
            if abundance <= min_abundance:
                continue
            if min_psm_threshold > 0 and taxon_psm_count[col] < min_psm_threshold:
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

        # Persist a single unified output TSV containing every available
        # abundance vector (PSM, biomass, cell-equivalent) plus the marker
        # / proteome-mass diagnostic columns. This replaces the previous
        # split between marker_correction_results.tsv and
        # biomass_correction_results.tsv.
        if output_dir and unified_table:
            unified_path = self._write_unified_results(
                taxon_labels=taxon_labels,
                taxon_names=[lbl.split("|", 1)[-1] for lbl in taxon_labels],
                psm_abundance=model.pi_,
                biomass_result=biomass_result,
                marker_result=marker_result,
                output_dir=output_dir,
                min_abundance=min_abundance,
                min_psm_threshold=min_psm_threshold,
                taxon_psm_count=taxon_psm_count,
            )

            # Diagnostic txt outputs (marker / biomass) live in diagnostics/.
            self._write_correction_diagnostics(
                marker_result=marker_result,
                biomass_result=biomass_result,
                output_dir=output_dir,
            )

            # Final step: render the bar-chart PNG.
            if generate_plot and unified_path is not None:
                from .abundance_em_core.visualize_results import plot_abundance_results

                plot_path = Path(str(output_dir)) / "abundance_plot.png"
                try:
                    plot_abundance_results(
                        unified_result_path=unified_path,
                        output_path=plot_path,
                        top_n=plot_top_n,
                    )
                except Exception as exc:  # noqa: BLE001 — never fail a run on a plot
                    logger.warning("abundance plot failed: %s", exc)

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

    # ----------------------------------------------------- marker correction

    @staticmethod
    def _run_marker_correction(model, mapping_result, spectral_counts, config):
        """Run hmmsearch + cell-equivalent correction.

        Returns a :class:`MarkerCorrectionResult`, or ``None`` if the
        prerequisites (HMM profile dir, hmmsearch binary, marker hits)
        are unmet — failures are logged at WARNING and never raise.
        """
        from .abundance_em_core.hmm_marker_search import run_hmmsearch
        from .abundance_em_core.marker_correction import (
            compute_cell_equivalent_abundance,
            log_marker_diagnostics,
        )

        hmm_profile_dir = config.get("hmm_profile_dir")
        if not hmm_profile_dir:
            logger.warning(
                "marker_correction enabled but hmm_profile_dir not set; "
                "skipping cell-equivalent correction"
            )
            return None

        try:
            marker_search = run_hmmsearch(
                fasta_path=str(config["fasta_path"]),
                hmm_profile_dir=str(hmm_profile_dir),
                evalue_threshold=float(config.get("marker_evalue", 1e-10)),
                cache_dir=str(config["output_dir"])
                if config.get("output_dir") else None,
                taxon_protein_peptides=mapping_result.taxon_protein_peptides,
                cpu=config.get("hmmsearch_cpu"),
            )
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("marker correction skipped: %s", exc)
            return None

        if not marker_search.marker_proteins:
            logger.warning(
                "hmmsearch returned no marker hits; skipping cell-equivalent "
                "correction"
            )
            return None

        # Spectral-count dict used by marker correction must align with
        # the y vector the EM saw.  Re-key by row order in case the
        # caller-supplied dict is out-of-sync (e.g. spectra without an
        # entry default to 0, which is what we want).
        sc_for_marker = dict(spectral_counts) if spectral_counts else {}
        if not sc_for_marker:
            sc_for_marker = {p: 1 for p in mapping_result.peptide_list}

        taxon_kingdom = AbundanceEMPlugin._build_taxon_kingdom(
            mapping_result.taxon_labels
        )

        result = compute_cell_equivalent_abundance(
            pi=model.pi_,
            responsibilities=model.responsibilities_,
            spectral_counts=sc_for_marker,
            mapping_matrix=mapping_result.matrix,
            taxon_labels=mapping_result.taxon_labels,
            peptide_index=mapping_result.peptide_index,
            marker_proteins=marker_search.marker_proteins,
            taxon_protein_peptides=mapping_result.taxon_protein_peptides,
            min_marker_families=int(config.get("min_marker_families", 3)),
            min_marker_psms=float(config.get("min_marker_psms", 1.0)),
            taxon_kingdom=taxon_kingdom,
        )

        log_marker_diagnostics(result, logger_obj=logger)
        return result

    @staticmethod
    def _build_taxon_kingdom(taxon_labels: list) -> dict:
        """Classify taxon labels into broad kingdoms using name heuristics.

        This is a coarse, keyword-based classifier tuned for the Kleiner
        FASTA layout (and similar mock-community databases).  Replace with
        a proper NCBI/GTDB taxonomy lookup if a general solution is needed.

        Rules (checked in order):
          1. "reinhardtii" or "chlamydomonas" in name → Eukaryota
          2. "phage", "virus", or "viridae" in name   → Virus
          3. "nitrososphaera", "viennensis",
             "thaumarchaeota", "euryarchaeota",
             "crenarchaeota" in name                  → Archaea
          4. Everything else                           → Bacteria

        Returns
        -------
        dict[str, str]
            ``taxon_label -> "Bacteria" | "Archaea" | "Eukaryota" | "Virus"``
        """
        kingdom_map: dict = {}
        for label in taxon_labels:
            name = label.split("|", 1)[-1] if "|" in label else label
            name_lower = name.lower()
            if any(kw in name_lower for kw in ("reinhardtii", "chlamydomonas")):
                kingdom = "Eukaryota"
            elif any(kw in name_lower for kw in ("phage", "virus", "viridae")):
                kingdom = "Virus"
            elif any(kw in name_lower for kw in (
                "nitrososphaera", "viennensis",
                "thaumarchaeota", "euryarchaeota", "crenarchaeota",
            )):
                kingdom = "Archaea"
            else:
                kingdom = "Bacteria"
            kingdom_map[label] = kingdom
        return kingdom_map

    @staticmethod
    def _taxon_psm_counts(A, y, responsibilities):
        """Per-taxon total expected PSM count.

        ``count_t = sum_p y_p * r_{pt}``: weights each peptide's spectral
        count by the EM responsibility assigned to taxon t. This is the
        natural fractional-PSM denominator for a min-PSM filter on
        shared-peptide samples.
        """
        import numpy as np

        y_arr = np.asarray(y, dtype=np.float64)
        r = np.asarray(responsibilities, dtype=np.float64)
        return (y_arr[:, np.newaxis] * r).sum(axis=0)

    # ------------------------------------------------- proteome-mass correction

    @staticmethod
    def _run_proteome_mass_correction(model, mapping_result):
        """Run proteome-size-weighted biomass correction.

        Returns a :class:`ProteomeMassCorrectionResult`, or ``None`` on failure
        (errors are logged at WARNING and never raised).
        """
        from .abundance_em_core.proteome_mass_correction import (
            compute_proteome_sizes,
            compute_biomass_abundance,
            log_proteome_mass_diagnostics,
        )

        try:
            proteome_sizes = compute_proteome_sizes(
                taxon_total_protein_counts=mapping_result.taxon_total_protein_counts,
                taxon_labels=mapping_result.taxon_labels,
            )
            biomass_result = compute_biomass_abundance(
                pi=model.pi_,
                proteome_sizes=proteome_sizes,
                taxon_labels=mapping_result.taxon_labels,
            )
            log_proteome_mass_diagnostics(biomass_result, logger=logger)
            return biomass_result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Proteome-mass correction failed: %s", exc)
            return None

    @staticmethod
    def _write_unified_results(
        taxon_labels: list,
        taxon_names: list,
        psm_abundance,
        biomass_result,
        marker_result,
        output_dir,
        min_abundance: float = 0.0,
        min_psm_threshold: int = 0,
        taxon_psm_count=None,
    ):
        """Write the single unified ``abundance_results.tsv``.

        Columns: ``taxon_id, taxon_name, psm_abundance, biomass_abundance,
        cell_abundance, proteome_size, marker_families, marker_psms,
        has_marker_estimate``.

        - ``biomass_abundance`` mirrors ``psm_abundance`` when proteome-mass
          correction was not run.
        - ``cell_abundance`` mirrors ``psm_abundance`` when marker correction
          was not run.
        - cRAP contaminant rows are dropped (defensive — they are excluded
          from the EM upstream, but a hand-edited mapping could in theory
          let one slip through).
        - Sorted by ``psm_abundance`` descending.
        """
        import numpy as np

        out_dir = Path(str(output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = out_dir / "abundance_results.tsv"

        psm = np.asarray(psm_abundance, dtype=np.float64)
        T = psm.shape[0]

        if biomass_result is not None:
            biomass = np.asarray(biomass_result.biomass_abundance, dtype=np.float64)
            proteome_sizes = np.asarray(biomass_result.proteome_sizes, dtype=np.int64)
        else:
            biomass = psm.copy()
            proteome_sizes = np.zeros(T, dtype=np.int64)

        if marker_result is not None:
            cell = np.asarray(marker_result.cell_abundance, dtype=np.float64)
            marker_families = np.asarray(
                marker_result.marker_families_per_taxon, dtype=np.int64
            )
            marker_psms = np.asarray(marker_result.marker_psm_count, dtype=np.float64)
            has_marker = np.asarray(marker_result.has_marker_estimate, dtype=bool)
        else:
            cell = psm.copy()
            marker_families = np.zeros(T, dtype=np.int64)
            marker_psms = np.zeros(T, dtype=np.float64)
            has_marker = np.zeros(T, dtype=bool)

        order = np.argsort(-psm)
        n_written = 0
        with tsv_path.open("w", encoding="utf-8") as fh:
            fh.write(
                "taxon_id\ttaxon_name\tpsm_abundance\tbiomass_abundance\t"
                "cell_abundance\tproteome_size\tmarker_families\t"
                "marker_psms\thas_marker_estimate\n"
            )
            for t in order:
                lbl = taxon_labels[t]
                tid, tname = lbl.split("|", 1) if "|" in lbl else ("0", lbl)
                # Drop cRAP rows — they should never be in the matrix, but
                # if a future change weakens the upstream filter we still
                # do not want them in the published table.
                if "crap" in str(tname).lower() or str(tid).lower().startswith("crap"):
                    continue
                if psm[t] <= min_abundance:
                    continue
                if (
                    min_psm_threshold > 0
                    and taxon_psm_count is not None
                    and taxon_psm_count[t] < min_psm_threshold
                ):
                    continue
                fh.write(
                    f"{tid}\t{tname}\t"
                    f"{psm[t]:.6f}\t"
                    f"{biomass[t]:.6f}\t"
                    f"{cell[t]:.6f}\t"
                    f"{int(proteome_sizes[t])}\t"
                    f"{int(marker_families[t])}\t"
                    f"{marker_psms[t]:.4f}\t"
                    f"{int(bool(has_marker[t]))}\n"
                )
                n_written += 1
        logger.info(
            "Unified abundance results: %s (%d rows)", tsv_path, n_written,
        )
        return tsv_path

    @staticmethod
    def _write_correction_diagnostics(
        marker_result,
        biomass_result,
        output_dir,
    ) -> None:
        """Write the human-readable correction reports under ``diagnostics/``.

        These are descriptive text files (not data); the per-taxon numeric
        breakdown lives in the unified ``abundance_results.tsv``.
        """
        diag_dir = Path(str(output_dir)) / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        if marker_result is not None:
            from .abundance_em_core.marker_correction import log_marker_diagnostics

            text = log_marker_diagnostics(marker_result)
            (diag_dir / "marker_diagnostics.txt").write_text(text, encoding="utf-8")

        if biomass_result is not None:
            from .abundance_em_core.proteome_mass_correction import (
                log_proteome_mass_diagnostics,
            )

            text = log_proteome_mass_diagnostics(biomass_result)
            (diag_dir / "biomass_diagnostics.txt").write_text(text, encoding="utf-8")
