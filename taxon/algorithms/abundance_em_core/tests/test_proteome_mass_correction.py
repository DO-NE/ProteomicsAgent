"""
Tests for proteome_mass_correction module.

Test cases:
1. Basic correction: 3 taxa with known pi and W_t, verify b_t = normalize(pi * W).
2. Multi-strain effect: two taxa with same-size genomes but one has 3x more
   proteins in FASTA (simulating multi-strain). Verify that taxon gets 3x boost.
3. Equal proteome sizes: if all W_t are equal, b_t should equal pi_t.
4. Zero proteome size: handle gracefully (W_t = 1 fallback), no crash.
5. Integration with MappingMatrixResult: verify compute_proteome_sizes()
   counts proteins correctly from taxon_protein_peptides structure.

Run from repo root with::

    python -m pytest taxon/algorithms/abundance_em_core/tests/test_proteome_mass_correction.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from taxon.algorithms.abundance_em_core.proteome_mass_correction import (
    ProteomeMassCorrectionResult,
    compute_biomass_abundance,
    compute_proteome_sizes,
    log_proteome_mass_diagnostics,
)


# ------------------------------------------------------------------ helpers

def _make_taxon_protein_peptides(
    taxa_protein_counts: dict[str, int],
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """Build a synthetic taxon_protein_peptides dict with dummy peptides.

    Parameters
    ----------
    taxa_protein_counts : {taxon_label: n_proteins}

    Returns
    -------
    (taxon_protein_peptides, taxon_labels)
    """
    taxon_protein_peptides: dict[str, dict[str, list[str]]] = {}
    taxon_labels: list[str] = []
    for label, n_proteins in taxa_protein_counts.items():
        taxon_labels.append(label)
        prot_map: dict[str, list[str]] = {}
        for i in range(n_proteins):
            acc = f"{label.split('|')[0]}_prot{i:04d}"
            prot_map[acc] = [f"PEPTIDE{i:04d}"]
        taxon_protein_peptides[label] = prot_map
    return taxon_protein_peptides, taxon_labels


# ------------------------------------------------------------------ tests

class TestComputeProteomeSizes:
    def test_basic_protein_count(self):
        """W_t should equal the number of protein keys per taxon."""
        tpp, labels = _make_taxon_protein_peptides({
            "LT2|Salmonella typhimurium": 100,
            "Cup|Cupriavidus metallidurans": 50,
            "Pae|Pseudomonas aeruginosa": 200,
        })
        sizes = compute_proteome_sizes(tpp, labels)
        assert sizes.shape == (3,)
        assert int(sizes[0]) == 100
        assert int(sizes[1]) == 50
        assert int(sizes[2]) == 200

    def test_zero_proteins_fallback(self):
        """A taxon with no proteins in tpp should get W_t = 1 (not 0)."""
        tpp = {"A|TaxonA": {"A_prot0001": ["PEP1"]}, "B|TaxonB": {}}
        labels = ["A|TaxonA", "B|TaxonB"]
        sizes = compute_proteome_sizes(tpp, labels)
        assert sizes[0] == 1
        assert sizes[1] == 1  # fallback from 0

    def test_missing_label_fallback(self):
        """If a label is not in tpp at all, W_t should be 1."""
        tpp = {"A|TaxonA": {"A_prot0001": ["PEP1"]}}
        labels = ["A|TaxonA", "B|TaxonB"]
        sizes = compute_proteome_sizes(tpp, labels)
        assert sizes[0] == 1
        assert sizes[1] == 1

    def test_ordering_matches_taxon_labels(self):
        """sizes[i] should correspond to taxon_labels[i], not insertion order of tpp."""
        tpp, _ = _make_taxon_protein_peptides({
            "X|TaxonX": 10,
            "Y|TaxonY": 30,
            "Z|TaxonZ": 5,
        })
        labels_reversed = ["Z|TaxonZ", "Y|TaxonY", "X|TaxonX"]
        sizes = compute_proteome_sizes(tpp, labels_reversed)
        assert int(sizes[0]) == 5   # Z
        assert int(sizes[1]) == 30  # Y
        assert int(sizes[2]) == 10  # X


class TestComputeBiomassAbundance:
    def test_basic_correction(self):
        """b_t = normalize(pi * W_t) for 3 taxa with known values."""
        pi = np.array([0.5, 0.3, 0.2])
        W = np.array([100.0, 200.0, 50.0])
        labels = ["A|TaxA", "B|TaxB", "C|TaxC"]

        result = compute_biomass_abundance(pi, W, labels)

        weighted = pi * W          # [50, 60, 10]
        expected = weighted / weighted.sum()   # [0.4167, 0.5, 0.0833]

        np.testing.assert_allclose(result.biomass_abundance, expected, atol=1e-10)
        np.testing.assert_allclose(result.biomass_abundance.sum(), 1.0, atol=1e-10)

    def test_equal_proteome_sizes_preserves_pi(self):
        """When all W_t are equal, b_t should equal π_t exactly."""
        pi = np.array([0.6, 0.25, 0.15])
        W = np.array([1000.0, 1000.0, 1000.0])
        labels = ["A|TaxA", "B|TaxB", "C|TaxC"]

        result = compute_biomass_abundance(pi, W, labels)

        np.testing.assert_allclose(result.biomass_abundance, pi, atol=1e-12)

    def test_multistrain_boost(self):
        """Taxon with 3x more proteins in FASTA gets 3x boost relative to the other.

        Two taxa with pi=[0.5, 0.5] but W=[3000, 1000].
        Expected: b = [0.75, 0.25].
        """
        pi = np.array([0.5, 0.5])
        W = np.array([3000.0, 1000.0])
        labels = ["Multi|SalmonellaLT2", "Single|EcoliK12"]

        result = compute_biomass_abundance(pi, W, labels)

        np.testing.assert_allclose(result.biomass_abundance[0], 0.75, atol=1e-10)
        np.testing.assert_allclose(result.biomass_abundance[1], 0.25, atol=1e-10)

    def test_result_fields_populated(self):
        """All dataclass fields on ProteomeMassCorrectionResult are set."""
        pi = np.array([0.4, 0.6])
        W = np.array([100.0, 200.0])
        labels = ["A|TaxA", "B|TaxB"]

        result = compute_biomass_abundance(pi, W, labels)

        assert result.n_taxa == 2
        assert result.min_proteome_size == 100
        assert result.max_proteome_size == 200
        assert result.median_proteome_size == 150.0
        np.testing.assert_array_equal(result.psm_abundance, pi)
        np.testing.assert_array_equal(result.proteome_sizes, W)
        np.testing.assert_array_equal(result.taxon_labels, labels)
        np.testing.assert_allclose(result.weighted_signal, pi * W)

    def test_zero_pi_taxon(self):
        """A taxon with pi=0 should also have b=0 regardless of W_t."""
        pi = np.array([1.0, 0.0])
        W = np.array([100.0, 99999.0])
        labels = ["A|TaxA", "B|TaxB"]

        result = compute_biomass_abundance(pi, W, labels)

        assert result.biomass_abundance[1] == pytest.approx(0.0)
        np.testing.assert_allclose(result.biomass_abundance.sum(), 1.0, atol=1e-10)

    def test_single_taxon(self):
        """With a single taxon, biomass_abundance should be [1.0]."""
        pi = np.array([1.0])
        W = np.array([500.0])
        labels = ["A|TaxA"]

        result = compute_biomass_abundance(pi, W, labels)

        np.testing.assert_allclose(result.biomass_abundance, [1.0], atol=1e-10)


class TestLogProteomeMassDiagnostics:
    def test_returns_string(self):
        """log_proteome_mass_diagnostics should return a non-empty string."""
        pi = np.array([0.5, 0.3, 0.2])
        W = np.array([100.0, 200.0, 50.0])
        labels = ["LT2|Salmonella typhimurium", "Cup|Cupriavidus metallidurans", "Pae|Pseudomonas aeruginosa"]
        result = compute_biomass_abundance(pi, W, labels)
        report = log_proteome_mass_diagnostics(result)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Proteome-Mass Correction" in report

    def test_report_contains_taxa_names(self):
        """The report should show taxon names from the labels."""
        pi = np.array([0.6, 0.4])
        W = np.array([1000.0, 500.0])
        labels = ["LT2|Salmonella typhimurium", "Cup|Cupriavidus metallidurans"]
        result = compute_biomass_abundance(pi, W, labels)
        report = log_proteome_mass_diagnostics(result)

        assert "Salmonella typhimurium" in report
        assert "Cupriavidus metallidurans" in report


class TestIntegrationWithTaxonProteinPeptides:
    def test_protein_count_not_peptide_count(self):
        """W_t counts protein accessions, not total peptides."""
        # 2 proteins, 5 peptides each → W_t = 2
        tpp = {
            "A|TaxA": {
                "prot_001": ["PEP1", "PEP2", "PEP3", "PEP4", "PEP5"],
                "prot_002": ["PEP6", "PEP7", "PEP8", "PEP9", "PEP10"],
            },
            # 10 proteins, 1 peptide each → W_t = 10
            "B|TaxB": {f"prot_{i:03d}": [f"PEP{100+i}"] for i in range(10)},
        }
        labels = ["A|TaxA", "B|TaxB"]
        sizes = compute_proteome_sizes(tpp, labels)

        assert int(sizes[0]) == 2
        assert int(sizes[1]) == 10

    def test_full_pipeline_compute_sizes_then_biomass(self):
        """End-to-end: build tpp, compute sizes, compute biomass."""
        tpp, labels = _make_taxon_protein_peptides({
            "LT2|Salmonella typhimurium": 500,
            "Cup|Cupriavidus metallidurans": 250,
        })
        pi = np.array([0.5, 0.5])

        sizes = compute_proteome_sizes(tpp, labels)
        result = compute_biomass_abundance(pi, sizes, labels)

        # W = [500, 250], pi = [0.5, 0.5]
        # weighted = [250, 125], normalized = [2/3, 1/3]
        np.testing.assert_allclose(result.biomass_abundance[0], 2 / 3, atol=1e-10)
        np.testing.assert_allclose(result.biomass_abundance[1], 1 / 3, atol=1e-10)
