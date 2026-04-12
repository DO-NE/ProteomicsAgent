"""Unit tests for the AbundanceEM core algorithm.

Run from the repository root with::

    python -m pytest taxon/algorithms/abundance_em_core/tests/ -v
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from taxon.algorithms.abundance_em_core.identifiability import identifiability_report
from taxon.algorithms.abundance_em_core.mapping_matrix import (
    _has_substring_match,
    _normalize_taxon_name,
    _parse_header,
    _should_exclude,
    build_mapping_matrix,
)
from taxon.algorithms.abundance_em_core.model import AbundanceEM
from taxon.algorithms.abundance_em_core.pepxml_parser import parse_pepxml
from taxon.algorithms.abundance_em_core.synthetic import (
    evaluate_recovery,
    generate_synthetic_community,
)


def _build_disjoint_matrix(true_pi: np.ndarray, peptides_per_taxon: int = 80):
    """Build a perfectly disjoint mapping matrix and a noiseless count vector."""
    T = true_pi.shape[0]
    P = T * peptides_per_taxon
    A = np.zeros((P, T), dtype=np.int8)
    for t in range(T):
        A[t * peptides_per_taxon : (t + 1) * peptides_per_taxon, t] = 1
    n_t = A.sum(axis=0)
    M = A.astype(float) / n_t
    phi = M @ true_pi
    rng = np.random.default_rng(0)
    y = rng.multinomial(50000, phi)
    return A, y


# ----------------------------------------------------------------------- tests


def test_perfect_recovery_no_sharing():
    """Disjoint peptide sets -> recovery should be near-exact."""
    true_pi = np.array([0.4, 0.25, 0.2, 0.1, 0.05])
    A, y = _build_disjoint_matrix(true_pi, peptides_per_taxon=80)

    model = AbundanceEM(alpha=1.0, max_iter=500, tol=1e-9, seed=0)
    model.fit(A, y)

    err = float(np.abs(model.pi_ - true_pi).sum())
    assert err < 0.01, f"L1 error too large: {err}"
    assert model.converged_


def test_recovery_with_sharing():
    """15% peptide sharing -> L1 error should be < 0.05."""
    data = generate_synthetic_community(
        n_taxa=5,
        n_peptides_per_taxon=200,
        shared_fraction=0.15,
        total_psms=15000,
        seed=11,
    )

    model = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-7, seed=0)
    model.fit(data["A"], data["y"])

    metrics = evaluate_recovery(data["true_pi"], model.pi_)
    assert metrics["l1_error"] < 0.05, metrics


def test_sparse_community():
    """20 taxa in DB, 3 truly present -> EM with sparse prior should detect them."""
    n_taxa = 20
    rng = np.random.default_rng(3)

    # Build a community where only 3 taxa have nonzero true abundance.
    true_pi = np.zeros(n_taxa)
    present = [2, 9, 17]
    true_pi[present] = [0.5, 0.3, 0.2]

    data = generate_synthetic_community(
        n_taxa=n_taxa,
        n_peptides_per_taxon=80,
        shared_fraction=0.10,
        true_pi=true_pi,
        total_psms=20000,
        seed=42,
    )

    model = AbundanceEM(
        alpha=0.5,
        max_iter=1000,
        tol=1e-8,
        n_restarts=3,
        min_abundance=1e-3,
        seed=0,
    )
    model.fit(data["A"], data["y"])

    detected = np.where(model.pi_ > 1e-3)[0].tolist()
    assert set(detected) == set(present), f"detected={detected}, expected={present}"
    # Mass on the present taxa should be close to 1.
    assert model.pi_[present].sum() > 0.99


def test_em_monotonicity():
    """Log-posterior should be non-decreasing across EM iterations."""
    data = generate_synthetic_community(
        n_taxa=6,
        n_peptides_per_taxon=150,
        shared_fraction=0.20,
        total_psms=8000,
        seed=5,
    )

    model = AbundanceEM(alpha=0.7, max_iter=300, tol=1e-9, seed=0)
    model.fit(data["A"], data["y"])

    history = model.log_posterior_history_
    assert len(history) >= 2
    diffs = np.diff(history)
    # Allow tiny floating-point regressions on the order of EPS * |lp|.
    assert (diffs >= -1e-7 * (np.abs(history[:-1]) + 1.0)).all(), (
        f"non-monotonic step: {diffs.min()}"
    )


def test_standard_errors():
    """Well-identified -> small SEs; poorly identified -> larger SEs."""
    # Well-identified: disjoint peptide sets.
    true_pi = np.array([0.5, 0.3, 0.2])
    A_clean, y_clean = _build_disjoint_matrix(true_pi, peptides_per_taxon=100)
    model_clean = AbundanceEM(alpha=1.0, max_iter=500, tol=1e-8, seed=0)
    model_clean.fit(A_clean, y_clean)

    # Poorly identified: heavy sharing.
    data_shared = generate_synthetic_community(
        n_taxa=3,
        n_peptides_per_taxon=120,
        shared_fraction=0.85,
        total_psms=5000,
        seed=21,
    )
    model_shared = AbundanceEM(alpha=1.0, max_iter=500, tol=1e-8, seed=0)
    model_shared.fit(data_shared["A"], data_shared["y"])

    assert model_clean.standard_errors_.max() < 0.05
    assert model_shared.standard_errors_.max() > model_clean.standard_errors_.max()


def test_identifiability_report():
    """Collinear taxa and unique-peptide counts should be reported correctly."""
    # 3 taxa: A and B share all peptides, C is disjoint.
    A = np.zeros((10, 3), dtype=np.int8)
    A[:5, 0] = 1
    A[:5, 1] = 1
    A[5:, 2] = 1

    report = identifiability_report(A, taxon_names=["A", "B", "C"])

    assert report["unique_peptide_counts"] == {"A": 0, "B": 0, "C": 5}
    assert "A" in report["at_risk_taxa"]
    assert "B" in report["at_risk_taxa"]
    assert "C" not in report["at_risk_taxa"]
    assert any(set(group) == {"A", "B"} for group in report["collinear_groups"])
    assert report["rank"] == 2
    assert report["identifiable"] is False


def test_dirichlet_prior_effect():
    """Sparse prior (alpha < 1) should suppress absent taxa more than alpha=1."""
    # Build a database with 10 taxa, only 3 truly present.
    n_taxa = 10
    true_pi = np.zeros(n_taxa)
    true_pi[[1, 4, 7]] = [0.5, 0.3, 0.2]
    data = generate_synthetic_community(
        n_taxa=n_taxa,
        n_peptides_per_taxon=100,
        shared_fraction=0.15,
        true_pi=true_pi,
        total_psms=8000,
        seed=99,
    )

    absent = [i for i in range(n_taxa) if i not in (1, 4, 7)]

    sparse_model = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-8, min_abundance=0.0, seed=0)
    sparse_model.fit(data["A"], data["y"])

    uniform_model = AbundanceEM(alpha=1.0, max_iter=500, tol=1e-8, min_abundance=0.0, seed=0)
    uniform_model.fit(data["A"], data["y"])

    sparse_absent = sparse_model.pi_[absent].sum()
    uniform_absent = uniform_model.pi_[absent].sum()
    assert sparse_absent <= uniform_absent + 1e-9, (
        f"sparse prior didn't suppress absent taxa: sparse={sparse_absent}, "
        f"uniform={uniform_absent}"
    )


def test_single_taxon():
    """Edge case: only one taxon -> pi == [1.0]."""
    A = np.ones((5, 1), dtype=np.int8)
    y = np.array([3, 1, 2, 0, 4])
    model = AbundanceEM(alpha=0.5, max_iter=10, tol=1e-8)
    model.fit(A, y)
    assert model.pi_.shape == (1,)
    assert pytest.approx(1.0, abs=1e-12) == float(model.pi_[0])
    assert model.converged_


def test_reproducibility_with_seed():
    """Same seed -> identical results."""
    data = generate_synthetic_community(
        n_taxa=4,
        n_peptides_per_taxon=120,
        shared_fraction=0.15,
        total_psms=6000,
        seed=33,
    )

    m1 = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-7, n_restarts=3, seed=2024)
    m1.fit(data["A"], data["y"])
    m2 = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-7, n_restarts=3, seed=2024)
    m2.fit(data["A"], data["y"])

    np.testing.assert_allclose(m1.pi_, m2.pi_, atol=1e-12)
    assert m1.n_iter_ == m2.n_iter_


# ------------------------------------------------------------------- header parsing


class TestParseHeader:
    """Tests for multi-format FASTA header parsing."""

    def test_uniprot_os_ox(self):
        hdr = (
            "tr|B5WWL1|B5WWL1_PSEAE Hypothetical membrane protein "
            "OS=Pseudomonas aeruginosa (strain ATCC 15692 / DSM 22644 / "
            "CIP 104116 / JCM 14847 / LMG 12228 / 1C / PRS 101 / PAO1) "
            "OX=208964 GN=PA0457.1 PE=4 SV=1"
        )
        tid, tname = _parse_header(hdr)
        assert tid == "208964"
        assert tname == "Pseudomonas aeruginosa"

    def test_uniprot_strain_normalization(self):
        hdr = "sp|P12345|X_ECOLI Something OS=Escherichia coli (strain K-12) OX=83333 GN=x PE=1 SV=1"
        _, tname = _parse_header(hdr)
        assert tname == "Escherichia coli"

    def test_genemark_species(self):
        hdr = (
            'gene_2|GeneMark.hmm|395_aa|+|1678|2865 >7e24d142adc2427d_1 '
            'assembly_id="7e24d142adc2427d" genome_id="79691302ed634fef" '
            'atcc_catalog_number="ATCC 43300" species="Staphylococcus aureus" '
            'contig_number="1" topology="circular"'
        )
        tid, tname = _parse_header(hdr)
        assert tname == "Staphylococcus aureus"
        assert tid != "0"  # should be a slug, not the default id

    def test_ncbi_bracket(self):
        hdr = "gi|123456|ref|NP_001234.1| some protein [Homo sapiens]"
        tid, tname = _parse_header(hdr)
        assert tname == "Homo sapiens"

    def test_empty_header(self):
        assert _parse_header("") == ("0", "unclassified")

    def test_no_match_returns_unclassified(self):
        tid, tname = _parse_header("SomeRandomHeader with no pattern")
        assert tname == "unclassified"


# --------------------------------------------------------------- taxon name normalization


class TestNormalizeTaxonName:
    def test_strip_strain(self):
        assert _normalize_taxon_name(
            "Pseudomonas aeruginosa (strain ATCC 15692)"
        ) == "Pseudomonas aeruginosa"

    def test_no_parens(self):
        assert _normalize_taxon_name("Homo sapiens") == "Homo sapiens"

    def test_empty(self):
        assert _normalize_taxon_name("") == ""


# --------------------------------------------------------------- exclude filtering


class TestShouldExclude:
    def test_decoy(self):
        assert _should_exclude("DECOY0_tr_1 Randomized", ["DECOY", "contag"])

    def test_contaminant(self):
        assert _should_exclude("contag|P0C1U8 something", ["DECOY", "contag"])

    def test_normal_uniprot(self):
        assert not _should_exclude(
            "tr|B5WWL1|B5WWL1_PSEAE Something OS=X OX=1", ["DECOY", "contag"]
        )

    def test_case_insensitive(self):
        assert _should_exclude("decoy_foo", ["DECOY"])

    def test_no_prefixes(self):
        assert not _should_exclude("DECOY_foo", None)
        assert not _should_exclude("DECOY_foo", [])


# --------------------------------------------------------------- substring matching


class TestSubstringMatch:
    def test_exact_substring(self):
        assert _has_substring_match("AGIVDEK", {"AGIVDEKR", "XXXX"}, min_ratio=0.7)

    def test_reverse_containment(self):
        assert _has_substring_match("AGIVDEKR", {"AGIVDEK", "XXXX"}, min_ratio=0.7)

    def test_too_short_ratio(self):
        # 3/10 = 0.3, below 0.7 threshold
        assert not _has_substring_match("AGI", {"AGIVDEKRXX"}, min_ratio=0.7)

    def test_no_match(self):
        assert not _has_substring_match("ZZZZZZZ", {"AAAAAAA"}, min_ratio=0.7)


# --------------------------------------------------------------- mapping matrix with filtering


class TestBuildMappingMatrixFiltering:
    """Integration test: multi-format FASTA with filtering."""

    _FASTA = (
        ">tr|A001|A001_KP Protein1 OS=Klebsiella pneumoniae OX=573 GN=x PE=1 SV=1\n"
        "AGIVDEKRPEPTIDER\n"
        ">gene_1|GeneMark.hmm|100_aa|+|1|300 species=\"Staphylococcus aureus\"\n"
        "LDNGGNTTQDNSKANOTHERR\n"
        ">DECOY0_tr_1 Randomized Protein Sequence\n"
        "AGIVDEKRPEPTIDER\n"
        ">contag|C001 Endoproteinase Glu-C\n"
        "AGIVDEKRPEPTIDER\n"
    )

    def _write_fasta(self, tmpdir):
        p = os.path.join(tmpdir, "test.fasta")
        with open(p, "w") as f:
            f.write(self._FASTA)
        return p

    def test_decoy_contaminant_excluded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = self._write_fasta(tmpdir)
            peptides = ["AGIVDEK", "LDNGGNTTQDNSK"]
            A, pep_list, taxon_labels = build_mapping_matrix(
                peptides=peptides,
                fasta_path=fasta,
                exclude_prefixes=["DECOY", "contag"],
            )
            # Only KP and SA taxa should survive.
            label_names = [lbl.split("|", 1)[1] for lbl in taxon_labels]
            assert "Klebsiella pneumoniae" in label_names
            assert "Staphylococcus aureus" in label_names
            assert len(taxon_labels) == 2

    def test_no_filter_includes_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = self._write_fasta(tmpdir)
            peptides = ["AGIVDEK"]
            _, _, taxon_labels = build_mapping_matrix(
                peptides=peptides, fasta_path=fasta,
            )
            # Without filtering, decoy and contaminant entries get taxon labels.
            assert len(taxon_labels) >= 2  # at least the decoy and one real taxon


# --------------------------------------------------------------- pepXML parser


class TestParsePepxml:
    """Tests for the pepXML parser."""

    _PEPXML = """\
<?xml version="1.0" encoding="UTF-8"?>
<msms_pipeline_analysis>
<msms_run_summary>
<spectrum_query spectrum="s1" start_scan="1" end_scan="1"
                assumed_charge="2" index="1">
  <search_result>
    <search_hit hit_rank="1" peptide="AGIVDEK"
                protein="tr|A001|A001_KP" num_tot_proteins="1">
    </search_hit>
  </search_result>
</spectrum_query>
<spectrum_query spectrum="s2" start_scan="2" end_scan="2"
                assumed_charge="2" index="2">
  <search_result>
    <search_hit hit_rank="1" peptide="LDNGGNTTQDNSK"
                protein="gene_1|GeneMark" num_tot_proteins="1">
    </search_hit>
  </search_result>
</spectrum_query>
<spectrum_query spectrum="s3" start_scan="3" end_scan="3"
                assumed_charge="2" index="3">
  <search_result>
    <search_hit hit_rank="1" peptide="AGIVDEK"
                protein="tr|A001|A001_KP" num_tot_proteins="2">
      <alternative_protein protein="DECOY0_tr_1"/>
    </search_hit>
  </search_result>
</spectrum_query>
<spectrum_query spectrum="s4" start_scan="4" end_scan="4"
                assumed_charge="2" index="4">
  <search_result>
    <search_hit hit_rank="1" peptide="DECOYONLY"
                protein="DECOY0_tr_2" num_tot_proteins="1">
    </search_hit>
  </search_result>
</spectrum_query>
<spectrum_query spectrum="s5" start_scan="5" end_scan="5"
                assumed_charge="2" index="5">
  <search_result>
    <search_hit hit_rank="2" peptide="RANK2PEP"
                protein="tr|A002|A002_KP" num_tot_proteins="1">
    </search_hit>
  </search_result>
</spectrum_query>
</msms_run_summary>
</msms_pipeline_analysis>
"""

    def _write_pepxml(self, tmpdir):
        p = os.path.join(tmpdir, "test.pepxml")
        with open(p, "w") as f:
            f.write(self._PEPXML)
        return p

    def test_spectral_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_pepxml(tmpdir)
            counts, pmap = parse_pepxml(path)
            # AGIVDEK appears in s1 and s3 (rank 1), so count = 2.
            assert counts["AGIVDEK"] == 2
            assert counts["LDNGGNTTQDNSK"] == 1

    def test_decoy_only_hit_excluded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_pepxml(tmpdir)
            counts, _ = parse_pepxml(path)
            # DECOYONLY maps only to a decoy protein -> excluded entirely.
            assert "DECOYONLY" not in counts

    def test_alternative_decoy_stripped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_pepxml(tmpdir)
            _, pmap = parse_pepxml(path)
            # AGIVDEK has primary=tr|A001 + alt=DECOY0_tr_1.
            # Only the non-decoy protein should survive.
            assert "DECOY0_tr_1" not in pmap["AGIVDEK"]
            assert "tr|A001|A001_KP" in pmap["AGIVDEK"]

    def test_rank2_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_pepxml(tmpdir)
            counts, _ = parse_pepxml(path)
            assert "RANK2PEP" not in counts

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_pepxml("/nonexistent/file.pepxml")
