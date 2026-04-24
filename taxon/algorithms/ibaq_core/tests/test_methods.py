"""Unit tests for iBAQ quantification methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxon.algorithms.ibaq_core.methods import (
    ibaq_em,
    ibaq_theoretical,
    raw_ibaq,
    raw_sum,
    top_n_proteins,
)
from taxon.algorithms.ibaq_core.fasta_utils import count_tryptic_peptides


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_df(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Create a PSM DataFrame from (peptide, protein_acc, species, intensity)."""
    return pd.DataFrame(rows, columns=["peptide", "protein_acc", "species", "intensity"])


# ── raw_sum ───────────────────────────────────────────────────────────────────

class TestRawSum:
    def test_known_proportions(self):
        df = _make_df([
            ("PEP1", "P1", "Species_A", 60.0),
            ("PEP2", "P2", "Species_A", 40.0),
            ("PEP3", "P3", "Species_B", 100.0),
        ])
        result = raw_sum(df)
        assert abs(result["Species_A"] - 0.5) < 1e-9
        assert abs(result["Species_B"] - 0.5) < 1e-9

    def test_single_species(self):
        df = _make_df([
            ("PEP1", "P1", "Only", 10.0),
            ("PEP2", "P2", "Only", 20.0),
        ])
        result = raw_sum(df)
        assert abs(result["Only"] - 1.0) < 1e-9

    def test_empty(self):
        df = _make_df([])
        assert raw_sum(df) == {}


# ── raw_ibaq ──────────────────────────────────────────────────────────────────

class TestRawIbaq:
    def test_protein_size_normalization(self):
        """A protein with 2 peptides and intensity 100 should have lower iBAQ
        than a protein with 1 peptide and intensity 100."""
        df = _make_df([
            # Species A: one protein, 2 peptides, total intensity 100
            ("PEP1", "P1", "A", 50.0),
            ("PEP2", "P1", "A", 50.0),
            # Species B: one protein, 1 peptide, intensity 100
            ("PEP3", "P2", "B", 100.0),
        ])
        result = raw_ibaq(df)
        # P1 iBAQ = 100/2 = 50, P2 iBAQ = 100/1 = 100 → B > A
        assert result["B"] > result["A"]
        assert abs(result["A"] + result["B"] - 1.0) < 1e-9

    def test_equal_proteins(self):
        df = _make_df([
            ("PEP1", "P1", "A", 50.0),
            ("PEP2", "P2", "B", 50.0),
        ])
        result = raw_ibaq(df)
        assert abs(result["A"] - 0.5) < 1e-9

    def test_empty(self):
        df = _make_df([])
        assert raw_ibaq(df) == {}


# ── ibaq_theoretical ──────────────────────────────────────────────────────────

class TestIbaqTheoretical:
    def test_theoretical_normalization(self):
        """A protein with more theoretical peptides should be down-weighted."""
        df = _make_df([
            ("PEP1", "P1", "A", 100.0),
            ("PEP2", "P2", "B", 100.0),
        ])
        # P1 has 10 theoretical peptides, P2 has 2
        theo = {"P1": 10, "P2": 2}
        result = ibaq_theoretical(df, theo)
        # P1 iBAQ = 100/10 = 10, P2 iBAQ = 100/2 = 50 → B > A
        assert result["B"] > result["A"]

    def test_fallback_count(self):
        df = _make_df([
            ("PEP1", "P_UNKNOWN", "A", 50.0),
        ])
        # P_UNKNOWN not in map → uses fallback
        result = ibaq_theoretical(df, {}, fallback_count=5)
        assert abs(result["A"] - 1.0) < 1e-9

    def test_empty(self):
        df = _make_df([])
        assert ibaq_theoretical(df, {}) == {}


# ── top_n_proteins ────────────────────────────────────────────────────────────

class TestTopNProteins:
    def test_selects_top_n(self):
        df = _make_df([
            ("P1", "ProtA", "X", 100.0),
            ("P2", "ProtB", "X", 50.0),
            ("P3", "ProtC", "X", 10.0),   # should be excluded with n=2
            ("P4", "ProtD", "Y", 80.0),
        ])
        result = top_n_proteins(df, n=2)
        # X uses ProtA(100) + ProtB(50) = 150, Y uses ProtD(80)
        assert result["X"] > result["Y"]
        total = result["X"] + result["Y"]
        assert abs(total - 1.0) < 1e-9

    def test_n_larger_than_proteins(self):
        """When n > actual protein count, all proteins are used."""
        df = _make_df([
            ("P1", "ProtA", "X", 100.0),
        ])
        result = top_n_proteins(df, n=10)
        assert abs(result["X"] - 1.0) < 1e-9

    def test_empty(self):
        df = _make_df([])
        assert top_n_proteins(df) == {}


# ── ibaq_em ───────────────────────────────────────────────────────────────────

class TestIbaqEm:
    def test_unique_peptides_only(self):
        """With no shared peptides, EM should behave like raw_ibaq."""
        df = _make_df([
            ("PEP1", "P1", "A", 60.0),
            ("PEP2", "P2", "B", 40.0),
        ])
        result_em = ibaq_em(df)
        result_raw = raw_ibaq(df)
        for sp in ("A", "B"):
            assert abs(result_em[sp] - result_raw[sp]) < 1e-6

    def test_shared_peptide_redistribution(self):
        """Shared peptides should be distributed proportionally to unique evidence."""
        df = _make_df([
            # Unique to A: intensity 80
            ("PEP1", "P1", "A", 80.0),
            # Unique to B: intensity 20
            ("PEP2", "P2", "B", 20.0),
            # Shared between A and B: intensity 100
            ("PEP3", "P1", "A", 100.0),
            ("PEP3", "P2", "B", 100.0),
        ])
        result = ibaq_em(df)
        # A has 80% of unique evidence, so should get ~80% of shared too
        assert result["A"] > result["B"]
        assert abs(result["A"] + result["B"] - 1.0) < 1e-6

    def test_all_shared(self):
        """When all peptides are shared, EM falls back to uniform initialization."""
        df = _make_df([
            ("PEP1", "P1", "A", 50.0),
            ("PEP1", "P2", "B", 50.0),
        ])
        result = ibaq_em(df)
        # No unique evidence → uniform → 50/50
        assert abs(result["A"] - 0.5) < 1e-6
        assert abs(result["B"] - 0.5) < 1e-6

    def test_convergence(self):
        """EM should converge (not raise or return empty)."""
        rows = []
        for i in range(20):
            sp = "A" if i < 15 else "B"
            rows.append((f"PEP{i}", f"P{i}", sp, float(i + 1)))
        # Add a few shared
        rows.append(("SHARED1", "P0", "A", 50.0))
        rows.append(("SHARED1", "P15", "B", 50.0))
        df = _make_df(rows)
        result = ibaq_em(df)
        assert len(result) == 2
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_empty(self):
        df = _make_df([])
        assert ibaq_em(df) == {}


# ── fasta_utils ───────────────────────────────────────────────────────────────

class TestCountTrypticPeptides:
    def test_simple_sequence(self):
        # MKAAAK has cleavage after K at positions 1 and 5 (M|K, K|A not KP)
        # Actually: M-K | A-A-A-K  (K at pos 1, K at pos 5)
        # sites = [0, 2, 6]
        # 0 missed: [0:2] len=2, [2:6] len=4
        # 1 missed: [0:6] len=6
        # With min_len=1, max_len=30 → 3 peptides
        count = count_tryptic_peptides("MKAAAK", min_len=1, max_len=30, missed_cleavages=1)
        assert count == 3

    def test_no_cleavage_sites(self):
        # No K/R → one peptide (the whole sequence)
        count = count_tryptic_peptides("AAAAAAA", min_len=1, max_len=30)
        assert count == 1

    def test_kp_rule(self):
        # KP should NOT be cleaved
        # AKPA → sites = [0, 4] (K at pos 1, but next is P → no cut)
        count = count_tryptic_peptides("AKPA", min_len=1, max_len=30)
        assert count == 1

    def test_empty(self):
        assert count_tryptic_peptides("") == 0
