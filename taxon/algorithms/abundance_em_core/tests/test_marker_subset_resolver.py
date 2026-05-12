"""Unit tests for :mod:`marker_subset_resolver`.

The fixtures are inline so the suite does NOT depend on the real bac120
/ ar53 HMM bundles or the bundled Nayfach-30 TSV — every input is
constructed in ``tmp_path`` at test time.

Run from the repository root with::

    python -m pytest taxon/algorithms/abundance_em_core/tests/test_marker_subset_resolver.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taxon.algorithms.abundance_em_core.marker_subset_resolver import (
    NayfachResolution,
    identify_ribo_subset,
    load_bac120_ar53_descriptions,
    resolve_nayfach30_to_bac120,
)


# --------------------------------------------------------------------- fixtures


FIXTURE_HMM = """\
HMMER3/f [3.3.2 | Nov 2020]
NAME  TIGR00001
ACC   TIGR00001
DESC  ribosomal protein L35Ae
LENG  62
HMM   ...
//
HMMER3/f [3.3.2 | Nov 2020]
NAME  TIGR00019
DESC  prolyl-tRNA synthetase
LENG  450
HMM   ...
//
HMMER3/f [3.3.2 | Nov 2020]
NAME  TIGR00115
DESC  translation initiation factor IF-2
LENG  650
HMM   ...
//
HMMER3/f [3.3.2 | Nov 2020]
NAME  TIGR00468
DESC  phenylalanyl-tRNA synthetase, alpha subunit
LENG  340
HMM   ...
//
HMMER3/f [3.3.2 | Nov 2020]
NAME  TIGR00472
DESC  phenylalanyl-tRNA synthetase, beta subunit
LENG  790
HMM   ...
//
HMMER3/f [3.3.2 | Nov 2020]
NAME  PF00164
DESC  Ribosomal protein S12/S23
LENG  120
HMM   ...
//
HMMER3/f [3.3.2 | Nov 2020]
NAME  PF00298
DESC  Ribosomal protein L11, RNA binding domain
LENG  70
HMM   ...
//
"""


NAYFACH_TSV_FIXTURE = "\n".join([
    "marker_id\tko_id\tdescription",
    "BA00005\tK02519\ttranslation initiation factor IF-2",
    "BA00020\tK01889\tphenylalanyl-tRNA synthetase alpha chain",
    "BA00013\tK01890\tphenylalanyl-tRNA synthetase beta chain",
    "BA00024\tK02867\tlarge subunit ribosomal protein L11",
    "BA00026\tK02950\tsmall subunit ribosomal protein S12",
    "BA00099\tK99999\tnonexistent test marker for unmatched case",
])


@pytest.fixture
def hmm_dir(tmp_path: Path) -> Path:
    """Write FIXTURE_HMM into a single concatenated *.hmm file."""
    p = tmp_path / "fixtures.hmm"
    p.write_text(FIXTURE_HMM, encoding="utf-8")
    return tmp_path


@pytest.fixture
def nayfach_tsv(tmp_path: Path) -> Path:
    p = tmp_path / "nayfach.tsv"
    p.write_text(NAYFACH_TSV_FIXTURE + "\n", encoding="utf-8")
    return p


@pytest.fixture
def descriptions(hmm_dir: Path) -> dict:
    return load_bac120_ar53_descriptions(hmm_dir)


# --------------------------------------------------------------------- tests
# ---- HMM header loader -------------------------------------------------


class TestLoadDescriptions:
    def test_load_descriptions_parses_concatenated_hmm(self, descriptions):
        # All 7 profiles in the concatenated bundle must be present.
        assert set(descriptions.keys()) == {
            "TIGR00001", "TIGR00019", "TIGR00115",
            "TIGR00468", "TIGR00472",
            "PF00164", "PF00298",
        }
        assert descriptions["TIGR00115"] == "translation initiation factor IF-2"
        assert descriptions["PF00298"] == "Ribosomal protein L11, RNA binding domain"

    def test_load_descriptions_stops_at_HMM_section(self, tmp_path: Path):
        # Inject a fake post-HMM "DESC" that should NOT be picked up,
        # since it lives inside the emission matrix region of the
        # previous record.
        adversarial = (
            "HMMER3/f [3.3.2 | Nov 2020]\n"
            "NAME  TEST_REAL\n"
            "DESC  real description\n"
            "HMM   A C D E\n"
            "DESC  FAKE post-matrix should be ignored\n"
            "NAME  FAKE_NAME_in_matrix\n"
            "//\n"
            "HMMER3/f [3.3.2 | Nov 2020]\n"
            "NAME  TEST_NEXT\n"
            "DESC  next profile\n"
            "HMM   A C D E\n"
            "//\n"
        )
        d = tmp_path / "adversarial.hmm"
        d.write_text(adversarial, encoding="utf-8")
        descs = load_bac120_ar53_descriptions(tmp_path)
        assert descs == {
            "TEST_REAL": "real description",
            "TEST_NEXT": "next profile",
        }


# ---- ribo subset --------------------------------------------------------


class TestRiboSubset:
    def test_identify_ribo_subset_includes_ribosomal_proteins(self, descriptions):
        ribo, audit = identify_ribo_subset(descriptions)
        # 4 ribosomal entries: TIGR00001 (L35Ae), PF00164 (S12/S23),
        # PF00298 (L11, RNA binding domain), and TIGR00472 also matches
        # the bare token "L11"-style regex? No — TIGR00472 is
        # phenylalanyl-tRNA so it should not match.  The token regex
        # \b[rsl]\d+[a-z]?\b would fire only if the DESC contained
        # something like "RpS12" or "L11"; the actual DESC strings here
        # all have "ribosomal protein" substring, so the substring match
        # is what selects them.
        assert "TIGR00001" in ribo
        assert "PF00164" in ribo
        assert "PF00298" in ribo
        # Audit list is sorted by accession.
        accs = [a for a, _d in audit]
        assert accs == sorted(accs)

    def test_identify_ribo_subset_excludes_synthetase(self, descriptions):
        ribo, _audit = identify_ribo_subset(descriptions)
        # Aminoacyl-tRNA synthetases are NOT ribosomal proteins.
        assert "TIGR00019" not in ribo  # prolyl-tRNA synthetase
        assert "TIGR00468" not in ribo  # PheRS alpha
        assert "TIGR00472" not in ribo  # PheRS beta

    def test_identify_ribo_subset_excludes_if2(self, descriptions):
        ribo, _audit = identify_ribo_subset(descriptions)
        # IF-2 is a translation factor, not a ribosomal protein.
        assert "TIGR00115" not in ribo


# ---- nayfach-30 resolution ---------------------------------------------


class TestNayfach30Resolution:
    def test_resolve_nayfach30_maps_ribosomal_by_suffix(self, descriptions, nayfach_tsv):
        res = resolve_nayfach30_to_bac120(nayfach_tsv, descriptions)
        # BA00024 = L11 -> PF00298 ("Ribosomal protein L11, RNA binding domain")
        # BA00026 = S12 -> PF00164 ("Ribosomal protein S12/S23")
        assert res.audit_map["BA00024"] == ["PF00298"]
        assert res.audit_map["BA00026"] == ["PF00164"]
        assert "PF00298" in res.subset
        assert "PF00164" in res.subset

    def test_resolve_nayfach30_maps_if2(self, descriptions, nayfach_tsv):
        res = resolve_nayfach30_to_bac120(nayfach_tsv, descriptions)
        assert res.audit_map["BA00005"] == ["TIGR00115"]
        assert "TIGR00115" in res.subset

    def test_resolve_nayfach30_maps_pherrs_subunits(self, descriptions, nayfach_tsv):
        res = resolve_nayfach30_to_bac120(nayfach_tsv, descriptions)
        assert res.audit_map["BA00020"] == ["TIGR00468"]  # alpha
        assert res.audit_map["BA00013"] == ["TIGR00472"]  # beta

    def test_resolve_nayfach30_reports_unmatched(self, descriptions, nayfach_tsv):
        res = resolve_nayfach30_to_bac120(nayfach_tsv, descriptions)
        assert "BA00099" in res.unmatched

    def test_resolve_nayfach30_subset_is_union(self, descriptions, nayfach_tsv):
        res = resolve_nayfach30_to_bac120(nayfach_tsv, descriptions)
        # 5 mapped Nayfach ids -> 5 distinct bac120 families.
        assert res.subset == {
            "TIGR00115",  # IF-2
            "TIGR00468",  # PheRS alpha
            "TIGR00472",  # PheRS beta
            "PF00298",    # L11
            "PF00164",    # S12
        }
        assert len(res.subset) == 5

    def test_resolve_nayfach30_returns_dataclass(self, descriptions, nayfach_tsv):
        # Sanity: type contract.
        res = resolve_nayfach30_to_bac120(nayfach_tsv, descriptions)
        assert isinstance(res, NayfachResolution)
        assert isinstance(res.subset, set)
        assert isinstance(res.audit_map, dict)
        assert isinstance(res.unmatched, list)
