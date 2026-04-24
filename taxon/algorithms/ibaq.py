"""Intensity-based absolute quantification (iBAQ) taxon inference plugin.

Wraps :mod:`taxon.algorithms.ibaq_core` so it can be discovered by the
ProteomicsAgent ``TaxonRegistry``.  Provides five quantification methods
selectable via the ``method`` config key.

Available methods
-----------------
- ``raw_sum``          — Baseline: sum intensities per species (no iBAQ).
- ``raw_ibaq``         — iBAQ normalized by observed peptide count per protein.
- ``ibaq_theoretical`` — iBAQ normalized by theoretical tryptic peptide count.
- ``top_n_proteins``   — Use only the N most intense proteins per species.
- ``ibaq_em``          — EM shared-peptide redistribution followed by iBAQ.
"""

from __future__ import annotations

import logging
from pathlib import Path

from taxon.base_plugin import TaxonPlugin, TaxonResult

from .ibaq_core.fasta_utils import (
    build_protein_species_map,
    build_theoretical_peptide_counts,
)
from .ibaq_core.methods import (
    ibaq_em,
    ibaq_theoretical,
    raw_ibaq,
    raw_sum,
    top_n_proteins,
)
from .ibaq_core.psm_table import build_psm_table

logger = logging.getLogger(__name__)

_VALID_METHODS = frozenset({
    "raw_sum",
    "raw_ibaq",
    "ibaq_theoretical",
    "top_n_proteins",
    "ibaq_em",
})


class IbaqPlugin(TaxonPlugin):
    """Species-level abundance estimation via iBAQ quantification.

    Required config keys
    --------------------
    fasta_path : str
        Path to the protein FASTA database used during the search.

    Optional config keys
    --------------------
    pepxml_path : str
        Path to a validated pepXML file.  When provided, peptide-protein
        mappings and spectral counts are extracted directly from the pepXML.
        Otherwise, mappings are inferred by in-silico digestion of the FASTA.
    method : str
        Quantification method (default ``"raw_ibaq"``).  One of:
        ``raw_sum``, ``raw_ibaq``, ``ibaq_theoretical``,
        ``top_n_proteins``, ``ibaq_em``.
    top_n : int
        N for ``top_n_proteins`` method (default ``3``).
    max_iter : int
        Maximum EM iterations for ``ibaq_em`` (default ``500``).
    tol : float
        EM convergence tolerance (default ``1e-8``).
    min_peptide_length, max_peptide_length : int
        Peptide length bounds for in-silico digestion (defaults ``7`` / ``30``).
    missed_cleavages : int
        Allowed missed cleavages (default ``2``).
    exclude_prefixes : list[str]
        Protein-accession prefixes to exclude (default ``["DECOY", "contag"]``).
    min_abundance : float
        Post-fit threshold; taxa below this are excluded from results
        (default ``1e-4``).
    spectral_counts : dict[str, int]
        Peptide -> PSM count.  Populated automatically by the orchestrator
        when a pepXML is available.
    """

    name = "ibaq"
    description = (
        "Intensity-based absolute quantification (iBAQ) for species-level "
        "abundance estimation.  Supports five methods: raw_sum, raw_ibaq, "
        "ibaq_theoretical, top_n_proteins, and ibaq_em (EM shared-peptide "
        "redistribution)."
    )
    requires_internet = False

    # ------------------------------------------------------------------ API

    def validate_config(self, config: dict) -> bool:
        """Check that ``fasta_path`` exists."""
        fasta_path = config.get("fasta_path")
        if not fasta_path:
            return False
        if not Path(str(fasta_path)).is_file():
            return False
        method = config.get("method", "raw_ibaq")
        if method not in _VALID_METHODS:
            logger.warning("Unknown iBAQ method '%s'; valid: %s", method, _VALID_METHODS)
            return False
        return True

    def run(self, peptides: list[str], config: dict) -> list[TaxonResult]:
        """Run iBAQ quantification and return TaxonResult records."""
        fasta_path = str(config["fasta_path"])
        method = str(config.get("method", "raw_ibaq"))
        exclude_prefixes = config.get("exclude_prefixes", ["DECOY", "contag"])
        min_abundance = float(config.get("min_abundance", 1e-4))
        min_pep_len = int(config.get("min_peptide_length", 7))
        max_pep_len = int(config.get("max_peptide_length", 30))
        missed_cleavages = int(config.get("missed_cleavages", 2))
        spectral_counts = config.get("spectral_counts") or {}

        # --- peptide-protein mapping (prefer pepXML, fall back to digestion) ---
        pepxml_path = config.get("pepxml_path")
        peptide_protein_map = None
        if pepxml_path and Path(str(pepxml_path)).is_file():
            try:
                from .abundance_em_core.pepxml_parser import parse_pepxml

                sc, peptide_protein_map = parse_pepxml(
                    str(pepxml_path), exclude_prefixes=exclude_prefixes,
                )
                # Use pepXML-derived spectral counts if the caller didn't
                # provide richer ones.
                if not spectral_counts:
                    spectral_counts = sc
                if not peptides:
                    peptides = list(sc.keys())
            except Exception:
                logger.warning(
                    "Failed to parse pepXML at %s; falling back to FASTA digestion",
                    pepxml_path,
                    exc_info=True,
                )

        if not peptides:
            logger.warning("IbaqPlugin: empty peptide list, returning []")
            return []

        # --- build protein-species map and PSM table ---
        protein_species_map = build_protein_species_map(
            fasta_path, exclude_prefixes=exclude_prefixes,
        )

        psm_df = build_psm_table(
            peptides=peptides,
            spectral_counts=spectral_counts,
            protein_species_map=protein_species_map,
            peptide_protein_map=peptide_protein_map,
            fasta_path=fasta_path,
            exclude_prefixes=exclude_prefixes,
            min_peptide_len=min_pep_len,
            max_peptide_len=max_pep_len,
            missed_cleavages=missed_cleavages,
        )

        if psm_df.empty:
            logger.warning("IbaqPlugin: PSM table is empty, returning []")
            return []

        # --- run selected method ---
        logger.info("Running iBAQ method: %s", method)
        abundances = self._dispatch(method, psm_df, config, fasta_path)

        if not abundances:
            logger.warning("IbaqPlugin: method '%s' returned no results", method)
            return []

        # --- convert to TaxonResult ---
        results: list[TaxonResult] = []
        for species_name, abundance in abundances.items():
            if abundance <= min_abundance:
                continue
            species_peptides = (
                psm_df[psm_df["species"] == species_name]["peptide"]
                .unique()
                .tolist()
            )
            results.append(
                TaxonResult(
                    taxon_id="0",
                    taxon_name=species_name,
                    rank="species",
                    abundance=abundance,
                    confidence=min(abundance * 10, 1.0),
                    peptide_count=len(species_peptides),
                    peptides=species_peptides,
                )
            )

        results.sort(key=lambda r: r.abundance, reverse=True)
        logger.info(
            "IbaqPlugin (%s): %d taxa above min_abundance=%.1e",
            method, len(results), min_abundance,
        )
        return results

    # --------------------------------------------------------------- dispatch

    def _dispatch(
        self,
        method: str,
        psm_df,
        config: dict,
        fasta_path: str,
    ) -> dict[str, float]:
        if method == "raw_sum":
            return raw_sum(psm_df)

        if method == "raw_ibaq":
            return raw_ibaq(psm_df)

        if method == "ibaq_theoretical":
            theo_counts = build_theoretical_peptide_counts(
                fasta_path,
                min_len=int(config.get("min_peptide_length", 7)),
                max_len=int(config.get("max_peptide_length", 30)),
                missed_cleavages=int(config.get("missed_cleavages", 2)),
                exclude_prefixes=config.get("exclude_prefixes", ["DECOY", "contag"]),
            )
            return ibaq_theoretical(psm_df, theo_counts)

        if method == "top_n_proteins":
            n = int(config.get("top_n", 3))
            return top_n_proteins(psm_df, n=n)

        if method == "ibaq_em":
            return ibaq_em(
                psm_df,
                max_iter=int(config.get("max_iter", 500)),
                tol=float(config.get("tol", 1e-8)),
            )

        return {}
