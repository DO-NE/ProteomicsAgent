"""Peptide detectability prediction for weighted EM emission.

Provides per-peptide detectability weights d_p that modify the emission model
from uniform to weighted:

    Current:   P(p|t) = A_{pt} / n_t                          (uniform)
    Weighted:  P(p|t) = d_p * A_{pt} / sum_{p'}(d_{p'} * A_{p't})  (weighted)

Two predictors are available:

- :class:`SequenceFeaturePredictor` — computes d_p from physicochemical
  sequence features (length, hydrophobicity, charge, missed cleavages,
  proline content) using a logistic combination. No external dependencies.
- :class:`DbyDeepPredictor` — loads pre-computed scores from a TSV file and
  falls back to :class:`SequenceFeaturePredictor` for unknown peptides.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DetectabilityPredictor(ABC):
    """Abstract interface for peptide detectability prediction.

    Subclasses must implement :meth:`predict`, which returns a score in
    ``(0, 1]`` for each peptide indicating its relative probability of
    being observed by the mass spectrometer.
    """

    @abstractmethod
    def predict(
        self,
        peptides: List[str],
        protein_sequences: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """Return detectability scores for each peptide.

        Parameters
        ----------
        peptides : list of str
            Peptide amino-acid sequences.
        protein_sequences : dict mapping protein accession to sequence, optional
            Parent protein sequences (unused by the base features predictor
            but available for subclasses that need protein-level context).

        Returns
        -------
        np.ndarray, shape ``(len(peptides),)``
            Detectability scores in ``(0, 1]``.
        """


class SequenceFeaturePredictor(DetectabilityPredictor):
    """Predict peptide detectability from physicochemical sequence features.

    Combines five features via multiplicative scoring:

    1. **Length** -- Gaussian window centred at 13 aa, favouring 8-18 aa.
       Peptides shorter than 6 or longer than 25 are penalised.
    2. **Hydrophobicity** -- Kyte-Doolittle GRAVY score, Gaussian centred
       at ~0.25 (moderately hydrophobic peptides ionise best in ESI).
    3. **Charge at pH ~2** -- counts K, R, H (half-weight) plus N-terminus.
       Charge 2-3 is optimal for tryptic peptides; charge > 4 is penalised.
    4. **Missed cleavages** -- internal K/R not at C-terminus; multiplicative
       penalty of 0.7 per missed cleavage site.
    5. **Proline content** -- mild bonus (proline aids CID fragmentation).

    All scores are floored at ``epsilon`` so no peptide receives zero weight.

    Parameters
    ----------
    epsilon : float, optional
        Minimum score floor (default ``0.01``).
    """

    # Kyte-Doolittle hydrophobicity scale.
    _KD_SCALE: Dict[str, float] = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
        "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
        "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "D": -3.5,
        "E": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5,
    }

    def __init__(self, epsilon: float = 0.01) -> None:
        if epsilon <= 0 or epsilon > 1:
            raise ValueError("epsilon must be in (0, 1]")
        self.epsilon = epsilon

    def predict(
        self,
        peptides: List[str],
        protein_sequences: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        scores = np.array(
            [self._score_peptide(p) for p in peptides], dtype=np.float64,
        )
        return np.maximum(scores, self.epsilon)

    def _score_peptide(self, seq: str) -> float:
        seq = seq.upper()
        n = len(seq)
        if n == 0:
            return self.epsilon

        # 1. Length: Gaussian window favouring 8-18 aa (centre=13, sigma=5).
        length_score = float(np.exp(-0.5 * ((max(n, 8) - 13.0) / 5.0) ** 2))
        if n < 6:
            length_score *= 0.3
        elif n > 25:
            length_score *= 0.4

        # 2. Hydrophobicity (GRAVY): Gaussian centred at 0.25, sigma=1.5.
        gravy = float(np.mean([self._KD_SCALE.get(aa, 0.0) for aa in seq]))
        hydro_score = float(np.exp(-0.5 * ((gravy - 0.25) / 1.5) ** 2))

        # 3. Charge at pH ~2: K, R fully protonated; H half-protonated; +1
        #    for N-terminus.
        charge = 1.0 + seq.count("K") + seq.count("R") + 0.5 * seq.count("H")
        if 2.0 <= charge <= 3.0:
            charge_score = 1.0
        elif charge < 2.0:
            charge_score = 0.6
        else:
            charge_score = max(0.3, 1.0 - 0.15 * (charge - 3.0))

        # 4. Missed cleavages: internal K/R (not at C-terminus).
        missed = sum(1 for aa in seq[:-1] if aa in ("K", "R"))
        mc_penalty = 0.7 ** missed

        # 5. Proline content: mild bonus (aids CID fragmentation).
        pro_fraction = seq.count("P") / n
        pro_bonus = 1.0 + 0.3 * min(pro_fraction, 0.15) / 0.15

        score = length_score * hydro_score * charge_score * mc_penalty * pro_bonus
        return float(np.clip(score, self.epsilon, 1.0))


class DbyDeepPredictor(DetectabilityPredictor):
    """Load pre-computed detectability scores from a TSV file.

    The TSV must have at least two columns: ``peptide_sequence`` and
    ``detectability_score`` (tab-separated, with a header row).  Peptides
    not found in the lookup table fall back to
    :class:`SequenceFeaturePredictor`.

    Parameters
    ----------
    tsv_path : str
        Path to the scores TSV file.
    epsilon : float, optional
        Minimum score floor (default ``0.01``).
    """

    def __init__(self, tsv_path: str, epsilon: float = 0.01) -> None:
        self.tsv_path = tsv_path
        self.epsilon = epsilon
        self._lookup = self._load_tsv(tsv_path)
        self._fallback = SequenceFeaturePredictor(epsilon=epsilon)

    @staticmethod
    def _load_tsv(path: str) -> Dict[str, float]:
        lookup: Dict[str, float] = {}
        with open(path, "r") as fh:
            header = fh.readline()  # skip header
            if not header:
                return lookup
            for line in fh:
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) >= 2:
                    try:
                        lookup[parts[0].strip()] = float(parts[1])
                    except ValueError:
                        continue
        return lookup

    def predict(
        self,
        peptides: List[str],
        protein_sequences: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        scores = np.empty(len(peptides), dtype=np.float64)
        n_lookup = 0
        n_fallback = 0

        for i, pep in enumerate(peptides):
            if pep in self._lookup:
                scores[i] = self._lookup[pep]
                n_lookup += 1
            else:
                scores[i] = self._fallback._score_peptide(pep)
                n_fallback += 1

        logger.info(
            "DbyDeepPredictor: %d/%d peptides from lookup, %d/%d from fallback",
            n_lookup, len(peptides), n_fallback, len(peptides),
        )

        return np.maximum(scores, self.epsilon)
