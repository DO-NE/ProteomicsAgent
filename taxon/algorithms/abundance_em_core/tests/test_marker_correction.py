"""Unit tests for the post-EM marker correction module.

These tests do NOT require HMMER to be installed: the hmmsearch step
is bypassed by hand-constructing the marker_proteins /
taxon_protein_peptides dicts that
:func:`taxon.algorithms.abundance_em_core.marker_correction.compute_cell_equivalent_abundance`
expects.

Run from the repository root with::

    python -m pytest taxon/algorithms/abundance_em_core/tests/test_marker_correction.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from taxon.algorithms.abundance_em_core.mapping_matrix import (
    MappingMatrixResult,
    build_mapping_matrix,
)
from taxon.algorithms.abundance_em_core.marker_correction import (
    MarkerCorrectionResult,
    compute_cell_equivalent_abundance,
    log_marker_diagnostics,
)


# --------------------------------------------------------------------- helpers


def _make_synthetic_inputs(
    pi: np.ndarray,
    spectra_per_peptide: dict,
    peptide_to_taxa: dict,
    taxa: list,
):
    """Hand-build (responsibilities, A, peptide_index, taxon_labels, sc).

    Responsibilities use the EM identity ``r_{pt} = pi_t M_{pt} / phi_p``
    on a uniform-emission mapping matrix, so the test inputs are
    self-consistent with the EM but bypass the iterative fit.
    """
    peptides = sorted(peptide_to_taxa.keys())
    P = len(peptides)
    T = len(taxa)
    label_to_idx = {lbl: i for i, lbl in enumerate(taxa)}

    A = np.zeros((P, T), dtype=np.int8)
    for i, pep in enumerate(peptides):
        for tlbl in peptide_to_taxa[pep]:
            A[i, label_to_idx[tlbl]] = 1

    n_t = A.sum(axis=0).astype(np.float64)
    n_t_safe = np.where(n_t == 0, 1.0, n_t)
    M = A.astype(np.float64) / n_t_safe
    phi = M @ pi
    phi = np.maximum(phi, 1e-12)
    R = (M * pi[np.newaxis, :]) / phi[:, np.newaxis]

    peptide_index = {p: i for i, p in enumerate(peptides)}
    sc = {p: float(spectra_per_peptide.get(p, 0)) for p in peptides}

    return R, A, peptide_index, peptides, sc


# --------------------------------------------------------------------- tests


class TestSyntheticThreeTaxa:
    """A clean 3-taxon sanity check.

    Setup:
      - 3 taxa, each with 5 marker proteins (one per family),
      - 1 unique marker peptide per (taxon, family) — total 15 peptides,
      - true cell counts equal across taxa, but proteome size and PSM
        signal differ so that PSM-level pi diverges from cell-equivalent c.
    """

    @pytest.fixture
    def setup(self):
        taxa = ["1|A", "2|B", "3|C"]
        # Each taxon has 5 marker proteins, each producing 1 unique marker
        # peptide.  Peptides named "<taxon>_<family>".
        families = ["F1", "F2", "F3", "F4", "F5"]

        # marker_proteins: protein_acc -> (taxon_label, family, e, score)
        marker_proteins: dict = {}
        # taxon_protein_peptides: taxon -> {acc -> [peps]}
        tpp: dict = {tlbl: {} for tlbl in taxa}
        # spectra_per_peptide: peptide -> count
        sc: dict = {}
        # peptide_to_taxa for matrix build
        pep_to_taxa: dict = {}

        # Use widely different per-taxon PSM weights so that PSM-pi
        # would put them at (0.6, 0.3, 0.1) while marker-c is (1/3 each).
        psm_weight = {"1|A": 60, "2|B": 30, "3|C": 10}

        for tlbl in taxa:
            for fam in families:
                acc = f"acc_{tlbl.split('|')[1]}_{fam}"
                pep = f"PEP_{tlbl.split('|')[1]}_{fam}"
                marker_proteins[acc] = (tlbl, fam, 1e-30, 200.0)
                tpp[tlbl][acc] = [pep]
                pep_to_taxa[pep] = {tlbl}
                # All marker peptides get equal share of taxon weight.
                sc[pep] = psm_weight[tlbl] / len(families)

        # Add some non-marker, taxon-unique peptides too, to give the EM
        # something extra to push pi away from the marker-c estimate.
        for tlbl in taxa:
            for k in range(20):
                pep = f"NONMARK_{tlbl.split('|')[1]}_{k}"
                pep_to_taxa[pep] = {tlbl}
                sc[pep] = psm_weight[tlbl] / 20.0

        # PSM-level pi — total signal proportional to psm_weight.
        total = sum(psm_weight.values())
        pi = np.array([psm_weight[t] / total for t in taxa])

        R, A, peptide_index, peptides, sc_dict = _make_synthetic_inputs(
            pi=pi,
            spectra_per_peptide=sc,
            peptide_to_taxa=pep_to_taxa,
            taxa=taxa,
        )

        return {
            "pi": pi,
            "responsibilities": R,
            "A": A,
            "peptide_index": peptide_index,
            "peptides": peptides,
            "spectral_counts": sc_dict,
            "taxa": taxa,
            "marker_proteins": marker_proteins,
            "taxon_protein_peptides": tpp,
        }

    def test_cell_abundance_equalises_when_marker_signal_equal(self, setup):
        """Marker peptides distributed equally across taxa => c ≈ uniform."""
        # Override sc so each taxon contributes the same total marker PSMs.
        sc = dict(setup["spectral_counts"])
        for k in list(sc.keys()):
            if k.startswith("PEP_"):
                sc[k] = 10.0

        result = compute_cell_equivalent_abundance(
            pi=setup["pi"],
            responsibilities=setup["responsibilities"],
            spectral_counts=sc,
            mapping_matrix=setup["A"],
            taxon_labels=setup["taxa"],
            peptide_index=setup["peptide_index"],
            marker_proteins=setup["marker_proteins"],
            taxon_protein_peptides=setup["taxon_protein_peptides"],
            min_marker_families=3,
            min_marker_psms=1.0,
        )

        # All three taxa should pass thresholds.
        assert result.has_marker_estimate.all()
        # And cell abundance should be close to 1/3 for each.
        np.testing.assert_allclose(
            result.cell_abundance, np.full(3, 1 / 3), atol=1e-6,
        )
        # PSM abundance is preserved as input.
        np.testing.assert_allclose(result.psm_abundance, setup["pi"])

    def test_marker_signal_differs_from_psm_pi(self, setup):
        """When marker PSMs are deliberately uneven, c reflects markers, not pi."""
        sc = dict(setup["spectral_counts"])
        # Make marker counts equal (5 per family, 5 families => 25/taxon)
        for k in list(sc.keys()):
            if k.startswith("PEP_"):
                sc[k] = 5.0

        result = compute_cell_equivalent_abundance(
            pi=setup["pi"],
            responsibilities=setup["responsibilities"],
            spectral_counts=sc,
            mapping_matrix=setup["A"],
            taxon_labels=setup["taxa"],
            peptide_index=setup["peptide_index"],
            marker_proteins=setup["marker_proteins"],
            taxon_protein_peptides=setup["taxon_protein_peptides"],
        )

        # PSM-pi was 0.6 / 0.3 / 0.1, but marker signal is now uniform,
        # so cell_abundance should be much closer to uniform than pi.
        assert result.has_marker_estimate.all()
        l1_pi = float(np.abs(setup["pi"] - 1 / 3).sum())
        l1_c = float(np.abs(result.cell_abundance - 1 / 3).sum())
        assert l1_c < l1_pi


class TestFallbackBehaviour:
    """Taxa that fail the family/PSM thresholds fall back to pi."""

    def test_taxon_with_zero_markers_uses_pi(self):
        taxa = ["1|A", "2|B"]
        # Only A has markers.
        marker_proteins = {
            "acc_A_F1": ("1|A", "F1", 1e-30, 100.0),
            "acc_A_F2": ("1|A", "F2", 1e-30, 100.0),
            "acc_A_F3": ("1|A", "F3", 1e-30, 100.0),
            "acc_A_F4": ("1|A", "F4", 1e-30, 100.0),
        }
        tpp = {
            "1|A": {
                "acc_A_F1": ["PEP_A_F1"],
                "acc_A_F2": ["PEP_A_F2"],
                "acc_A_F3": ["PEP_A_F3"],
                "acc_A_F4": ["PEP_A_F4"],
            },
            "2|B": {},
        }
        pep_to_taxa = {
            "PEP_A_F1": {"1|A"},
            "PEP_A_F2": {"1|A"},
            "PEP_A_F3": {"1|A"},
            "PEP_A_F4": {"1|A"},
            "B_OTHER1": {"2|B"},
            "B_OTHER2": {"2|B"},
        }
        sc = {
            "PEP_A_F1": 10.0, "PEP_A_F2": 10.0,
            "PEP_A_F3": 10.0, "PEP_A_F4": 10.0,
            "B_OTHER1": 25.0, "B_OTHER2": 25.0,
        }
        pi = np.array([0.4, 0.6])  # arbitrary
        R, A, peptide_index, peptides, sc_dict = _make_synthetic_inputs(
            pi=pi, spectra_per_peptide=sc,
            peptide_to_taxa=pep_to_taxa, taxa=taxa,
        )

        result = compute_cell_equivalent_abundance(
            pi=pi,
            responsibilities=R,
            spectral_counts=sc_dict,
            mapping_matrix=A,
            taxon_labels=taxa,
            peptide_index=peptide_index,
            marker_proteins=marker_proteins,
            taxon_protein_peptides=tpp,
            min_marker_families=3,
            min_marker_psms=1.0,
        )

        # A passed (4 families), B did not (zero markers).
        assert bool(result.has_marker_estimate[0])
        assert not bool(result.has_marker_estimate[1])
        # B fell back to its pi value (after final renormalisation).
        # The renorm: A gets 1.0 of marker share, B gets pi_B = 0.6.
        # Total = 1.6, so c_A = 1/1.6 = 0.625, c_B = 0.6/1.6 = 0.375.
        np.testing.assert_allclose(
            result.cell_abundance, [1 / 1.6, 0.6 / 1.6], atol=1e-6,
        )
        # cell_abundance still sums to 1.
        assert abs(result.cell_abundance.sum() - 1.0) < 1e-9

    def test_taxon_with_too_few_families_falls_back(self):
        taxa = ["1|A"]
        # Only 2 marker families — below default threshold of 3.
        marker_proteins = {
            "acc_A_F1": ("1|A", "F1", 1e-30, 100.0),
            "acc_A_F2": ("1|A", "F2", 1e-30, 100.0),
        }
        tpp = {"1|A": {"acc_A_F1": ["P1"], "acc_A_F2": ["P2"]}}
        pep_to_taxa = {"P1": {"1|A"}, "P2": {"1|A"}}
        sc = {"P1": 5.0, "P2": 5.0}
        pi = np.array([1.0])
        R, A, peptide_index, _peps, sc_dict = _make_synthetic_inputs(
            pi, sc, pep_to_taxa, taxa,
        )

        result = compute_cell_equivalent_abundance(
            pi=pi, responsibilities=R, spectral_counts=sc_dict,
            mapping_matrix=A, taxon_labels=taxa,
            peptide_index=peptide_index,
            marker_proteins=marker_proteins,
            taxon_protein_peptides=tpp,
            min_marker_families=3,
        )

        assert not bool(result.has_marker_estimate[0])
        np.testing.assert_allclose(result.cell_abundance, pi)


class TestSharedMarkerPeptide:
    """Shared marker peptides should split signal via EM responsibilities."""

    def test_shared_peptide_splits_per_responsibility(self):
        # Two taxa share every marker peptide.  pi defines the split:
        # if pi = (0.7, 0.3) and a peptide can come from either, then
        # under uniform emission its responsibility is also (0.7, 0.3)
        # because both proteomes have the same single peptide for it.
        taxa = ["1|A", "2|B"]
        # 3 marker families, all peptides shared.
        fams = ["F1", "F2", "F3"]
        marker_proteins: dict = {}
        tpp: dict = {"1|A": {}, "2|B": {}}
        pep_to_taxa: dict = {}
        sc: dict = {}
        for f in fams:
            pep = f"SHARED_{f}"
            for tlbl in taxa:
                acc = f"acc_{tlbl.split('|')[1]}_{f}"
                marker_proteins[acc] = (tlbl, f, 1e-30, 100.0)
                tpp[tlbl][acc] = [pep]
            pep_to_taxa[pep] = set(taxa)
            sc[pep] = 30.0  # 30 PSMs per shared peptide

        pi = np.array([0.7, 0.3])
        R, A, peptide_index, _peps, sc_dict = _make_synthetic_inputs(
            pi, sc, pep_to_taxa, taxa,
        )

        result = compute_cell_equivalent_abundance(
            pi=pi, responsibilities=R, spectral_counts=sc_dict,
            mapping_matrix=A, taxon_labels=taxa,
            peptide_index=peptide_index,
            marker_proteins=marker_proteins,
            taxon_protein_peptides=tpp,
            min_marker_families=3,
            min_marker_psms=1.0,
        )

        # Marker signal distribution should mirror pi (since responsibilities
        # are pi-weighted under uniform emission).
        assert result.has_marker_estimate.all()
        np.testing.assert_allclose(result.cell_abundance, pi, atol=1e-6)
        # Total marker PSMs == 90 (3 peptides × 30 PSMs).
        assert abs(result.total_marker_psms - 90.0) < 1e-6


class TestMappingMatrixResult:
    """Verify MappingMatrixResult.taxon_protein_peptides is consistent with A."""

    _FASTA = (
        ">prot1 OS=Bacillus subtilis OX=1423 GN=test\n"
        "MAGEKAGIVDEKLIVDEKLIVDEKLIVDEKLIVDEKLR\n"
        ">prot2 OS=Bacillus subtilis OX=1423 GN=test\n"
        "MGGGAGKLIVDEKAGIVDEKLDNGGNTTQDNSKR\n"
        ">prot3 OS=Escherichia coli OX=562 GN=test\n"
        "MGEKLDNGGNTTQDNSKLDNGGNTTQDNSKLDNGGNTTQDNSKR\n"
    )

    def test_taxon_protein_peptides_population(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = Path(tmpdir) / "tiny.fasta"
            fasta.write_text(self._FASTA)
            peptides = ["AGIVDEK", "LIVDEK", "LDNGGNTTQDNSK"]
            result = build_mapping_matrix(
                peptides=peptides,
                fasta_path=str(fasta),
                missed_cleavages=0,
                min_length=6,
            )

            assert isinstance(result, MappingMatrixResult)
            assert result.matrix.shape[0] == len(peptides)
            assert len(result.taxon_labels) >= 1
            # Every taxon column should have at least one source protein.
            for label in result.taxon_labels:
                assert label in result.taxon_protein_peptides, label
                prot_map = result.taxon_protein_peptides[label]
                assert prot_map  # non-empty
                # Every recorded peptide must also produce a 1 in the matrix.
                col = result.taxon_labels.index(label)
                for acc, peps in prot_map.items():
                    assert all(p in result.peptide_index for p in peps)
                    for p in peps:
                        row = result.peptide_index[p]
                        assert result.matrix[row, col] == 1, (
                            f"Inconsistent: {p} attributed to {acc} in {label} "
                            f"but A[{row},{col}] = 0"
                        )

    def test_backwards_compatible_unpacking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = Path(tmpdir) / "tiny.fasta"
            fasta.write_text(self._FASTA)
            peptides = ["AGIVDEK", "LIVDEK", "LDNGGNTTQDNSK"]
            # Old 4-tuple unpack pattern must still work.
            A, peplist, labels, unclassified = build_mapping_matrix(
                peptides=peptides,
                fasta_path=str(fasta),
                missed_cleavages=0,
                min_length=6,
            )
            assert A.shape[0] == len(peplist)
            assert isinstance(labels, list)
            assert isinstance(unclassified, list)


class TestMarkerPeptideTable:
    """Per-marker-peptide diagnostic TSV (Cycle 6).

    These tests cover the new ``diagnostics/marker_peptides.tsv`` dump
    emitted by :func:`compute_cell_equivalent_abundance` when given a
    ``marker_peptide_table_path``.
    """

    _EXPECTED_HEADER = (
        "taxon_id\ttaxon_name\tmarker_family\tmarker_protein_accession\t"
        "marker_protein_hmm_evalue\tpeptide_sequence\tpeptide_length\t"
        "is_unique_peptide\tn_taxa_sharing\ty_p\tr_pt\t"
        "weighted_psm_contribution"
    )

    def _build_two_taxon_setup(self):
        """2 taxa × 5 marker proteins × 2 peptides each = 20 marker peptides."""
        taxa = ["1|A", "2|B"]
        families = ["F1", "F2", "F3", "F4", "F5"]

        marker_proteins: dict = {}
        tpp: dict = {tlbl: {} for tlbl in taxa}
        pep_to_taxa: dict = {}
        sc: dict = {}
        for tlbl in taxa:
            short = tlbl.split("|")[1]
            for fam in families:
                acc = f"acc_{short}_{fam}"
                marker_proteins[acc] = (tlbl, [fam], 1e-30, 200.0)
                p1 = f"PEP_{short}_{fam}_A"
                p2 = f"PEP_{short}_{fam}_B"
                tpp[tlbl][acc] = [p1, p2]
                pep_to_taxa[p1] = {tlbl}
                pep_to_taxa[p2] = {tlbl}
                sc[p1] = 6.0
                sc[p2] = 4.0

        pi = np.array([0.5, 0.5])
        R, A, peptide_index, _peps, sc_dict = _make_synthetic_inputs(
            pi, sc, pep_to_taxa, taxa,
        )
        return {
            "pi": pi, "responsibilities": R, "A": A,
            "peptide_index": peptide_index, "spectral_counts": sc_dict,
            "taxa": taxa, "marker_proteins": marker_proteins,
            "taxon_protein_peptides": tpp,
        }

    def test_marker_peptide_tsv_emitted(self):
        """File is created with the expected header and row count."""
        setup = self._build_two_taxon_setup()

        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "diagnostics" / "marker_peptides.tsv"
            result = compute_cell_equivalent_abundance(
                pi=setup["pi"],
                responsibilities=setup["responsibilities"],
                spectral_counts=setup["spectral_counts"],
                mapping_matrix=setup["A"],
                taxon_labels=setup["taxa"],
                peptide_index=setup["peptide_index"],
                marker_proteins=setup["marker_proteins"],
                taxon_protein_peptides=setup["taxon_protein_peptides"],
                min_marker_families=3,
                min_marker_psms=1.0,
                marker_peptide_table_path=str(tsv_path),
            )

            assert tsv_path.is_file(), f"TSV not created at {tsv_path}"

            lines = tsv_path.read_text(encoding="utf-8").splitlines()
            assert lines, "TSV is empty"
            assert lines[0] == self._EXPECTED_HEADER, (
                f"Header mismatch:\n  got: {lines[0]}\n  exp: {self._EXPECTED_HEADER}"
            )
            # 2 taxa × 5 families × 2 peptides = 20 rows; one row per
            # (taxon, family, protein, peptide).
            assert len(lines) - 1 == 20, (
                f"Expected 20 data rows, got {len(lines) - 1}"
            )

            # Both taxa qualify under default thresholds.
            assert result.has_marker_estimate.all()

    def test_marker_peptide_tsv_consistency(self):
        """Per-taxon dedup-summed contribution == marker_psm_count."""
        setup = self._build_two_taxon_setup()

        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "marker_peptides.tsv"
            result = compute_cell_equivalent_abundance(
                pi=setup["pi"],
                responsibilities=setup["responsibilities"],
                spectral_counts=setup["spectral_counts"],
                mapping_matrix=setup["A"],
                taxon_labels=setup["taxa"],
                peptide_index=setup["peptide_index"],
                marker_proteins=setup["marker_proteins"],
                taxon_protein_peptides=setup["taxon_protein_peptides"],
                marker_peptide_table_path=str(tsv_path),
            )

            # Read TSV, dedup by (taxon_id, peptide), sum per taxon.
            per_taxon_psms: dict = {}
            seen_pairs: set = set()
            with tsv_path.open("r", encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                col = {name: i for i, name in enumerate(header)}
                for raw in fh:
                    fields = raw.rstrip("\n").split("\t")
                    tid = fields[col["taxon_id"]]
                    pep = fields[col["peptide_sequence"]]
                    contrib = float(fields[col["weighted_psm_contribution"]])
                    if (tid, pep) in seen_pairs:
                        continue
                    seen_pairs.add((tid, pep))
                    per_taxon_psms[tid] = per_taxon_psms.get(tid, 0.0) + contrib

            # Compare per-taxon sums to result.marker_psm_count.
            label_to_idx = {lbl: i for i, lbl in enumerate(setup["taxa"])}
            for tlbl, t_idx in label_to_idx.items():
                tid = tlbl.split("|", 1)[0]
                expected = float(result.marker_psm_count[t_idx])
                got = per_taxon_psms.get(tid, 0.0)
                assert abs(got - expected) < 1e-6, (
                    f"Taxon {tlbl}: TSV sum {got} vs marker_psm_count "
                    f"{expected} (delta {got - expected:.3e})"
                )

    def test_unique_flag_correctness(self):
        """is_unique_peptide reflects A.sum(axis=1) == 1, not degeneracy."""
        # P1 maps only to T1; P2 maps to T1 AND T2.  Both are markers in T1.
        taxa = ["1|A", "2|B"]
        marker_proteins = {
            # T1 owns the protein that contains both P1 and P2.
            "acc_A_F1": ("1|A", ["F1"], 1e-30, 100.0),
            "acc_A_F2": ("1|A", ["F2"], 1e-30, 100.0),
            "acc_A_F3": ("1|A", ["F3"], 1e-30, 100.0),
        }
        tpp = {
            "1|A": {
                "acc_A_F1": ["P1"],            # unique to T1
                "acc_A_F2": ["P2"],            # shared with T2
                "acc_A_F3": ["P3UNIQUE"],     # unique to T1, separate fam
            },
            "2|B": {},
        }
        # Critical: P2 must appear in T2's tryptic digest so A[P2,:] = [1,1].
        pep_to_taxa = {
            "P1": {"1|A"},
            "P2": {"1|A", "2|B"},
            "P3UNIQUE": {"1|A"},
            # Give T2 a couple of non-marker peptides so its column is non-empty.
            "B_OTHER1": {"2|B"},
            "B_OTHER2": {"2|B"},
        }
        sc = {"P1": 8.0, "P2": 8.0, "P3UNIQUE": 8.0,
              "B_OTHER1": 5.0, "B_OTHER2": 5.0}
        pi = np.array([0.6, 0.4])

        R, A, peptide_index, _peps, sc_dict = _make_synthetic_inputs(
            pi, sc, pep_to_taxa, taxa,
        )

        # Sanity-check the synthetic A before running correction.
        assert A[peptide_index["P1"]].sum() == 1
        assert A[peptide_index["P2"]].sum() == 2

        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "marker_peptides.tsv"
            compute_cell_equivalent_abundance(
                pi=pi,
                responsibilities=R,
                spectral_counts=sc_dict,
                mapping_matrix=A,
                taxon_labels=taxa,
                peptide_index=peptide_index,
                marker_proteins=marker_proteins,
                taxon_protein_peptides=tpp,
                min_marker_families=3,
                min_marker_psms=1.0,
                marker_peptide_table_path=str(tsv_path),
            )

            # Parse the TSV and check is_unique_peptide / n_taxa_sharing.
            uniq_by_pep: dict = {}
            sharing_by_pep: dict = {}
            with tsv_path.open("r", encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                col = {name: i for i, name in enumerate(header)}
                for raw in fh:
                    fields = raw.rstrip("\n").split("\t")
                    pep = fields[col["peptide_sequence"]]
                    uniq = fields[col["is_unique_peptide"]] == "True"
                    ns = int(fields[col["n_taxa_sharing"]])
                    uniq_by_pep[pep] = uniq
                    sharing_by_pep[pep] = ns

            assert uniq_by_pep["P1"] is True, "P1 should be unique"
            assert sharing_by_pep["P1"] == 1
            assert uniq_by_pep["P2"] is False, "P2 should not be unique"
            assert sharing_by_pep["P2"] == 2
            assert uniq_by_pep["P3UNIQUE"] is True
            assert sharing_by_pep["P3UNIQUE"] == 1


class TestDiagnosticReport:
    """log_marker_diagnostics smoke test — must produce a non-empty string."""

    def test_diagnostics_output_format(self):
        result = MarkerCorrectionResult(
            cell_abundance=np.array([0.6, 0.4]),
            psm_abundance=np.array([0.5, 0.5]),
            marker_signal=np.array([12.0, 8.0]),
            marker_families_per_taxon=np.array([5, 4]),
            marker_psm_count=np.array([12.0, 8.0]),
            marker_peptides_per_taxon=np.array([5, 4]),
            taxon_labels=["1|A", "2|B"],
            has_marker_estimate=np.array([True, True]),
            total_marker_psms=20.0,
            total_marker_peptides=9,
            fraction_psms_from_markers=0.2,
        )
        report = log_marker_diagnostics(result)
        assert "Marker-based Cell-Equivalent" in report
        assert "1|A".split("|")[1] in report  # "A"
        assert "marker" in report.lower()
