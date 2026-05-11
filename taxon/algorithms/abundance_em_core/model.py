"""Multinomial-mixture EM (MAP) for taxon abundance estimation.

The model treats observed peptide spectral counts as draws from a mixture of
taxon-specific peptide emission distributions, where each taxon emits its
member peptides with equal probability (uniform-emission baseline). Inference
is performed by Expectation-Maximization on the log-posterior with a Dirichlet
prior — symmetric by default, or an asymmetric empirical-Bayes prior anchored
on the PSM-weighted unique-peptide vector.

Generative model
----------------
- A in {0, 1}^{P x T} : peptide-to-taxon mapping (1 if taxon t can produce
  peptide p).
- n_t = sum_p A_{pt} : repertoire size of taxon t.
- M_{pt} = A_{pt} / n_t : per-taxon peptide emission probability.
- pi in simplex^{T-1} : taxon abundance vector (the inference target).
- phi_p(pi) = sum_t pi_t * M_{pt} : marginal probability of peptide p.
- y ~ Multinomial(N, phi(pi)).
- pi ~ Dirichlet(a) prior, where a is either:
    * symmetric:        a_t = alpha for every t (sparsity-inducing if < 1).
    * empirical_bayes:  a_t = prior_kappa * pi_hat_unique[t] + prior_alpha0,
                        with pi_hat_unique built from the PSM-weighted unique
                        peptides (see ``_compute_unique_psm_vector``).

EM updates
----------
E-step:
    r_{pt} = (pi_t * M_{pt}) / phi_p
    c_{pt} = y_p * r_{pt}

M-step (MAP with Dirichlet(a)):
    pi_t^new = (sum_p c_{pt} + a_t - 1) / (N + sum_t (a_t - 1))

In the symmetric mode ``a_t = alpha`` and the denominator reduces to
``N + T * (alpha - 1)``; in the empirical-Bayes mode it reduces to
``N + prior_kappa + T * (prior_alpha0 - 1)``. Components that go non-positive
(possible when any a_t < 1) are clamped to a small floor and the vector is
renormalized.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import linalg as sla

logger = logging.getLogger(__name__)

# Numerical floor used wherever a probability or denominator could be zero.
_EPS = 1e-12


@dataclass
class _FitState:
    """Internal container for the best run across restarts."""

    pi: np.ndarray
    log_posterior: float
    log_posterior_history: list
    n_iter: int
    converged: bool


class AbundanceEM:
    """Multinomial-mixture EM for taxon abundance estimation.

    Parameters
    ----------
    alpha : float, optional
        Dirichlet prior hyperparameter (default ``0.5``, Jeffreys prior).
        Values ``< 1`` encourage sparsity; ``alpha = 1`` is a uniform prior;
        ``alpha > 1`` smooths toward uniform abundances.
    max_iter : int, optional
        Maximum EM iterations per restart (default ``500``).
    tol : float, optional
        Convergence tolerance on the L1 norm of the change in pi
        (default ``1e-6``).
    n_restarts : int, optional
        Number of random Dirichlet(1) restarts. The fit with the highest
        log-posterior is kept (default ``1``).
    min_abundance : float, optional
        Post-convergence threshold. Taxa with ``pi < min_abundance`` are zeroed
        and the vector is renormalized (default ``1e-4``).
    init : {"unique", "unique_psm", "uniform", "random"}, optional
        Initialization strategy for the first run (default ``"unique"``).
        ``"unique"`` weights each taxon by the **count** of unique peptides
        with nonzero spectral counts; ``"unique_psm"`` weights each taxon by
        the **sum of spectral counts** over its unique peptides (so a taxon
        with two heavily-observed unique peptides outranks one with five
        barely-observed unique peptides); ``"uniform"`` uses 1/T; ``"random"``
        samples from Dirichlet(1, ..., 1). Restarts always use random.
    seed : int or None, optional
        Random seed for reproducibility.
    prior_mode : {"symmetric", "empirical_bayes"}, optional
        Form of the Dirichlet prior (default ``"symmetric"``).
        ``"symmetric"`` uses ``Dir(alpha, ..., alpha)`` — identical to the
        legacy behavior. ``"empirical_bayes"`` uses an asymmetric
        ``Dir(a_1, ..., a_T)`` with
        ``a_t = prior_kappa * pi_hat_unique[t] + prior_alpha0``, where
        ``pi_hat_unique`` is the PSM-weighted unique-peptide vector
        (see ``_compute_unique_psm_vector``). ``alpha`` is ignored in this
        mode; ``prior_alpha0`` provides the symmetric baseline and
        ``prior_kappa`` controls how strongly the prior is pulled toward the
        unique-peptide signal.
    prior_kappa : float, optional
        Concentration of the empirical-Bayes prior. Larger values pull the
        M-step output more strongly toward ``pi_hat_unique``. Defaults to
        ``0.0`` (no pull; with ``prior_alpha0 == alpha`` the EB mode collapses
        to the symmetric mode). Ignored when ``prior_mode='symmetric'``.
    prior_alpha0 : float, optional
        Symmetric baseline of the empirical-Bayes prior (default ``0.5``).
        Only consulted when ``prior_mode='empirical_bayes'``.
    detectability_mode : {"uniform", "sequence_features", "file"}, optional
        How to compute per-peptide detectability weights (default
        ``"uniform"``).  ``"uniform"`` reproduces the original unweighted
        emission model.  ``"sequence_features"`` uses physicochemical
        features via :class:`SequenceFeaturePredictor`.  ``"file"`` loads
        pre-computed scores via :class:`DbyDeepPredictor`.
    detectability_file : str or None, optional
        Path to a TSV file with pre-computed detectability scores (required
        when ``detectability_mode='file'`` unless ``detectability_weights``
        is given directly).
    detectability_weights : np.ndarray or None, optional
        Direct injection of a ``(P,)`` weight vector.  When provided this
        overrides the mode-based computation.

    Attributes
    ----------
    pi_ : np.ndarray, shape ``(T,)``
        Estimated abundance vector after fitting.
    responsibilities_ : np.ndarray, shape ``(P, T)``
        Final E-step responsibilities ``r_{pt}``.
    log_posterior_history_ : list of float
        Log-posterior at each iteration of the best run (length = n_iter_).
    converged_ : bool
        Whether EM converged within ``max_iter``.
    n_iter_ : int
        Number of iterations executed in the best run.
    standard_errors_ : np.ndarray, shape ``(T,)``
        Approximate standard errors from the observed Fisher information.
    fisher_singular_ : bool
        True if the observed information matrix was rank-deficient and a
        pseudoinverse was used.
    A_ : np.ndarray, shape ``(P, T)``
        Binary mapping matrix stored after ``fit()``.
    M_ : np.ndarray, shape ``(P, T)``
        Uniform emission matrix stored after ``fit()``.
    W_ : np.ndarray, shape ``(P, T)``
        Emission matrix used in EM (detectability-weighted, or equal to M_
        when ``detectability_mode='uniform'``).
    peptide_list_ : list of str or None
        Peptide sequences (row labels), if passed to ``fit()``.
    taxon_labels_ : list of str or None
        Taxon labels (column labels), if passed to ``fit()``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        max_iter: int = 500,
        tol: float = 1e-6,
        n_restarts: int = 1,
        min_abundance: float = 1e-4,
        init: str = "unique",
        seed: Optional[int] = None,
        detectability_mode: str = "uniform",
        detectability_file: Optional[str] = None,
        detectability_weights: Optional[np.ndarray] = None,
        prior_mode: str = "symmetric",
        prior_kappa: float = 0.0,
        prior_alpha0: float = 0.5,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if tol <= 0:
            raise ValueError("tol must be > 0")
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        if init not in ("unique", "unique_psm", "uniform", "random"):
            raise ValueError(
                "init must be 'unique', 'unique_psm', 'uniform', or 'random'"
            )
        if detectability_mode not in ("uniform", "sequence_features", "file"):
            raise ValueError(
                "detectability_mode must be 'uniform', 'sequence_features', "
                "or 'file'"
            )
        if (
            detectability_mode == "file"
            and detectability_file is None
            and detectability_weights is None
        ):
            raise ValueError(
                "detectability_file is required when detectability_mode='file'"
            )
        if prior_mode not in ("symmetric", "empirical_bayes"):
            raise ValueError(
                "prior_mode must be 'symmetric' or 'empirical_bayes'"
            )
        if prior_kappa < 0:
            raise ValueError("prior_kappa must be >= 0")
        if prior_alpha0 <= 0:
            raise ValueError("prior_alpha0 must be > 0")

        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_restarts = int(n_restarts)
        self.min_abundance = float(min_abundance)
        self.init = init
        self.seed = seed
        self.detectability_mode = detectability_mode
        self.detectability_file = detectability_file
        self.detectability_weights = (
            np.asarray(detectability_weights, dtype=np.float64)
            if detectability_weights is not None
            else None
        )
        self.prior_mode = prior_mode
        self.prior_kappa = float(prior_kappa)
        self.prior_alpha0 = float(prior_alpha0)

        # Set after fit().
        self.pi_: Optional[np.ndarray] = None
        self.responsibilities_: Optional[np.ndarray] = None
        self.log_posterior_history_: list = []
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.standard_errors_: Optional[np.ndarray] = None
        self.fisher_singular_: bool = False

        # Public post-fit matrix attributes.
        self.A_: Optional[np.ndarray] = None
        self.M_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.peptide_list_: Optional[list] = None
        self.taxon_labels_: Optional[list] = None

        # Cached internals (not part of public API).
        self._A: Optional[np.ndarray] = None
        self._M: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._taxon_names: Optional[list] = None
        self._W: Optional[np.ndarray] = None
        # Per-taxon Dirichlet concentration vector ``a_t`` used in the M-step.
        # Built once per ``fit`` call after the matrices are known. In the
        # symmetric mode this is the constant ``alpha`` vector; in the
        # empirical-Bayes mode it is ``prior_kappa * pi_hat_unique + prior_alpha0``.
        self._prior_alpha_vec: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ public

    def fit(
        self,
        A: np.ndarray,
        y: np.ndarray,
        peptide_sequences: Optional[list] = None,
        taxon_labels: Optional[list] = None,
    ) -> "AbundanceEM":
        """Fit the model.

        Parameters
        ----------
        A : np.ndarray, shape ``(P, T)``
            Peptide-to-taxon mapping matrix. Boolean or integer values are
            both accepted; non-binary entries are coerced to binary by
            comparing to zero.
        y : np.ndarray, shape ``(P,)``
            Spectral count vector (non-negative).
        peptide_sequences : list of str or None, optional
            Peptide amino-acid sequences, required when
            ``detectability_mode='sequence_features'``. If ``None`` and mode
            is not ``'uniform'``, falls back to uniform emission with a
            warning (unless ``detectability_weights`` was provided directly).
            Stored in ``self.peptide_list_`` after fitting.
        taxon_labels : list of str or None, optional
            Taxon column labels in the same order as columns of A.
            Stored in ``self.taxon_labels_`` after fitting.

        Returns
        -------
        self : AbundanceEM
            Fitted model. The estimated abundances live in ``self.pi_``.
        """
        A_arr = np.asarray(A)
        y_arr = np.asarray(y, dtype=np.float64)

        if A_arr.ndim != 2:
            raise ValueError("A must be 2-D with shape (P, T)")
        if y_arr.ndim != 1:
            raise ValueError("y must be 1-D with shape (P,)")
        if A_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"A and y must agree on the peptide axis "
                f"(A has {A_arr.shape[0]} rows, y has length {y_arr.shape[0]})"
            )
        if (y_arr < 0).any():
            raise ValueError("y must be non-negative")

        P, T = A_arr.shape
        if T == 0:
            raise ValueError("A must have at least one taxon column")

        # Coerce to binary float matrix once.
        A_bin = (A_arr != 0).astype(np.float64)
        n_t = A_bin.sum(axis=0)  # repertoire sizes

        if (n_t == 0).any():
            empty = np.where(n_t == 0)[0].tolist()
            logger.warning(
                "Dropping %d taxon column(s) with empty peptide repertoire: %s",
                len(empty),
                empty,
            )
        # Replace zero repertoires with 1 so M is well-defined; the
        # corresponding column of M is identically zero, so these taxa never
        # collect responsibility and the M-step pushes them to the prior.
        n_t_safe = np.where(n_t == 0, 1.0, n_t)
        M = A_bin / n_t_safe[np.newaxis, :]
        # Force the original empty columns of M back to zero so they cannot
        # collect any peptide responsibility.
        if (n_t == 0).any():
            M[:, n_t == 0] = 0.0

        # Build the (possibly detectability-weighted) emission matrix.
        W = self._build_emission_matrix(A_bin, M, peptide_sequences)

        # Build the per-taxon Dirichlet concentration vector ``a_t``.  In the
        # symmetric mode this is a constant ``alpha`` vector (so the M-step
        # and log-posterior reduce exactly to the legacy expressions); in the
        # empirical-Bayes mode it is anchored on the PSM-weighted unique
        # vector.  Computed ONCE so the EM loop and log-posterior share a
        # consistent prior.
        self._prior_alpha_vec = self._build_prior_alpha_vec(A_bin, y_arr, T)

        # Edge case: T == 1 forces pi = [1.0]; skip EM entirely.
        if T == 1:
            pi = np.array([1.0])
            responsibilities = np.zeros((P, 1), dtype=np.float64)
            mask = y_arr > 0
            responsibilities[mask, 0] = 1.0
            self._A = A_bin
            self._M = M
            self._W = W
            self._y = y_arr
            self.A_ = A_bin
            self.M_ = M
            self.W_ = W
            self.peptide_list_ = list(peptide_sequences) if peptide_sequences is not None else None
            self.taxon_labels_ = list(taxon_labels) if taxon_labels is not None else None
            self.pi_ = pi
            self.responsibilities_ = responsibilities
            self.log_posterior_history_ = [self._log_posterior(pi, W, y_arr)]
            self.converged_ = True
            self.n_iter_ = 0
            self.standard_errors_ = self._compute_standard_errors(pi, W, y_arr)
            return self

        rng = np.random.default_rng(self.seed)

        best: Optional[_FitState] = None
        for restart in range(self.n_restarts):
            if restart == 0:
                pi0 = self._initial_pi(self.init, T, A_bin, y_arr, rng)
            else:
                pi0 = self._initial_pi("random", T, A_bin, y_arr, rng)
            state = self._run_em(pi0, W, y_arr)
            logger.info(
                "EM restart %d/%d: %d iters, log-posterior=%.6f, converged=%s",
                restart + 1,
                self.n_restarts,
                state.n_iter,
                state.log_posterior,
                state.converged,
            )
            if best is None or state.log_posterior > best.log_posterior:
                best = state

        assert best is not None  # n_restarts >= 1, loop runs at least once

        pi = best.pi.copy()
        # Apply min_abundance threshold and renormalize.
        if self.min_abundance > 0:
            below = pi < self.min_abundance
            if below.any():
                pi[below] = 0.0
                total = pi.sum()
                if total > 0:
                    pi = pi / total
                else:  # pathological: everything fell below threshold
                    pi = np.full(T, 1.0 / T)

        responsibilities = self._responsibilities(pi, W)

        self._A = A_bin
        self._M = M
        self._W = W
        self._y = y_arr
        self.A_ = A_bin
        self.M_ = M
        self.W_ = W
        self.peptide_list_ = list(peptide_sequences) if peptide_sequences is not None else None
        self.taxon_labels_ = list(taxon_labels) if taxon_labels is not None else None
        self.pi_ = pi
        self.responsibilities_ = responsibilities
        self.log_posterior_history_ = best.log_posterior_history
        self.converged_ = best.converged
        self.n_iter_ = best.n_iter
        self.standard_errors_ = self._compute_standard_errors(pi, W, y_arr)
        return self

    def predict(self) -> np.ndarray:
        """Return the estimated abundance vector ``pi``.

        Returns
        -------
        np.ndarray, shape ``(T,)``
            Estimated taxon abundances. Sums to 1.

        Raises
        ------
        RuntimeError
            If the model has not yet been fit.
        """
        if self.pi_ is None:
            raise RuntimeError("AbundanceEM has not been fit yet; call fit() first")
        return self.pi_.copy()

    def get_results_dict(self, taxon_names: list) -> list:
        """Return per-taxon results as a list of dicts.

        Parameters
        ----------
        taxon_names : list of str
            Names for each taxon column, in the same order as ``A``.

        Returns
        -------
        list of dict
            One dict per taxon with keys ``taxon_name``, ``abundance``,
            ``confidence``, ``peptide_count``, ``peptides``. Sorted by
            abundance descending.
        """
        if self.pi_ is None or self.responsibilities_ is None:
            raise RuntimeError("AbundanceEM has not been fit yet; call fit() first")
        if len(taxon_names) != self.pi_.shape[0]:
            raise ValueError(
                f"taxon_names length ({len(taxon_names)}) does not match "
                f"number of taxa ({self.pi_.shape[0]})"
            )

        confidences = self._confidences()
        # Hard-assign each peptide to a taxon if its responsibility exceeds 0.5.
        # The same peptide can match at most one taxon under that rule because
        # responsibilities for a peptide row sum to <= 1.
        hard = self.responsibilities_ > 0.5

        results = []
        for t, name in enumerate(taxon_names):
            peptide_idx = np.where(hard[:, t])[0].tolist()
            results.append(
                {
                    "taxon_name": name,
                    "abundance": float(self.pi_[t]),
                    "confidence": float(confidences[t]),
                    "peptide_count": len(peptide_idx),
                    "peptides": peptide_idx,
                }
            )
        results.sort(key=lambda r: r["abundance"], reverse=True)
        return results

    # ----------------------------------------------------------------- internals

    def _build_emission_matrix(
        self,
        A_bin: np.ndarray,
        M: np.ndarray,
        peptide_sequences: Optional[list],
    ) -> np.ndarray:
        """Compute the emission matrix, optionally weighted by detectability.

        When ``detectability_mode='uniform'`` (and no direct weights are
        injected), the unweighted matrix ``M`` is returned unchanged.
        Otherwise, per-peptide detectability scores are used to build a
        column-normalised weighted emission matrix ``W``.
        """
        # Fast path: uniform mode with no override weights.
        if self.detectability_mode == "uniform" and self.detectability_weights is None:
            return M

        # Determine the weight vector d.
        if self.detectability_weights is not None:
            d = self.detectability_weights
        elif self.detectability_mode == "sequence_features":
            if peptide_sequences is None:
                logger.warning(
                    "detectability_mode='sequence_features' but no peptide "
                    "sequences provided; falling back to uniform emission"
                )
                return M
            from .detectability import SequenceFeaturePredictor

            d = SequenceFeaturePredictor().predict(peptide_sequences)
        elif self.detectability_mode == "file":
            if self.detectability_file is None:
                logger.warning(
                    "detectability_mode='file' but no file path provided; "
                    "falling back to uniform emission"
                )
                return M
            from .detectability import DbyDeepPredictor

            d = DbyDeepPredictor(self.detectability_file).predict(
                peptide_sequences or []
            )
        else:
            return M

        P = A_bin.shape[0]
        if d.shape[0] != P:
            raise ValueError(
                f"detectability weights length ({d.shape[0]}) must match "
                f"number of peptides ({P})"
            )

        # W_{pt} = d_p * A_{pt} / sum_{p'}(d_{p'} * A_{p't})
        dA = A_bin * d[:, np.newaxis]
        col_sums = dA.sum(axis=0)
        col_sums_safe = np.where(col_sums == 0, 1.0, col_sums)
        W = dA / col_sums_safe[np.newaxis, :]
        W[:, col_sums == 0] = 0.0

        logger.info(
            "Applied detectability weights: min=%.4f, max=%.4f, mean=%.4f",
            float(d.min()),
            float(d.max()),
            float(d.mean()),
        )
        return W

    @staticmethod
    def _compute_unique_psm_vector(A: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-taxon sum of PSM counts over peptides that are unique to it.

        A peptide ``p`` is "unique to taxon ``t``" when its row of ``A`` is
        non-zero only in column ``t`` (``sum_{t'} A[p, t'] == 1`` and
        ``A[p, t] == 1``). For each taxon this returns ``sum_{p in U_t} y_p``
        — the raw, unnormalised PSM-weighted unique-peptide signal. Callers
        that need a probability vector should add an epsilon and normalise.

        Both the ``init='unique_psm'`` initialiser and the empirical-Bayes
        prior anchor share this computation so they cannot drift apart.

        Parameters
        ----------
        A : np.ndarray, shape ``(P, T)``
            Binary mapping matrix (any non-zero entry is treated as 1).
        y : np.ndarray, shape ``(P,)``
            Spectral count vector.

        Returns
        -------
        np.ndarray, shape ``(T,)``
            ``sum_{p in U_t} y_p`` for each taxon ``t``. Non-negative.
        """
        A_bin = (np.asarray(A) != 0).astype(np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        # ``unique_mask[p] == True`` iff peptide p maps to exactly one taxon.
        unique_mask = (A_bin.sum(axis=1) == 1.0)
        if not unique_mask.any():
            return np.zeros(A_bin.shape[1], dtype=np.float64)
        # Restrict to unique peptides, then (y * column) summed per taxon
        # gives sum_{p in U_t} y_p because A_bin[p, t] is 0/1 on that slice.
        y_weighted = (y_arr * unique_mask)
        return (A_bin * y_weighted[:, np.newaxis]).sum(axis=0)

    @staticmethod
    def _initial_pi(
        strategy: str,
        T: int,
        A: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Build an initial abundance vector under the requested strategy."""
        if strategy == "uniform":
            return np.full(T, 1.0 / T)
        if strategy == "random":
            sample = rng.dirichlet(np.ones(T))
            return sample
        if strategy == "unique":
            # Count, per taxon, the number of peptides that (a) the taxon can
            # produce and (b) appear in the data with nonzero count.
            mask = y > 0
            weights = (A[mask, :] > 0).sum(axis=0).astype(np.float64)
            weights = weights + 1e-3  # epsilon avoids zero rows
            return weights / weights.sum()
        if strategy == "unique_psm":
            # PSM-weighted variant: sum of y_p over unique peptides instead
            # of the raw count.  Uses the same helper as the EB prior so the
            # two cannot diverge.
            weights = AbundanceEM._compute_unique_psm_vector(A, y)
            weights = weights + 1e-3  # matches the eps used by "unique"
            return weights / weights.sum()
        raise ValueError(f"unknown init strategy: {strategy}")

    def _run_em(
        self,
        pi0: np.ndarray,
        M: np.ndarray,
        y: np.ndarray,
    ) -> _FitState:
        """Run EM from a single starting point."""
        pi = pi0.astype(np.float64).copy()
        history = []
        converged = False
        n_iter = 0

        prev_lp = self._log_posterior(pi, M, y)
        history.append(prev_lp)

        for it in range(1, self.max_iter + 1):
            n_iter = it
            pi_new = self._em_step(pi, M, y)
            lp = self._log_posterior(pi_new, M, y)
            history.append(lp)

            # Monotonicity guard: if the log-posterior dropped (numerical
            # noise or a non-positive fixup), reject the step. The EM update
            # is theoretically non-decreasing, so this should be rare.
            if lp + 1e-9 < prev_lp:
                logger.debug(
                    "EM step %d not monotonic (lp %.9e -> %.9e); reverting",
                    it,
                    prev_lp,
                    lp,
                )
                history[-1] = prev_lp
                converged = True
                break

            delta = float(np.abs(pi_new - pi).sum())
            pi = pi_new
            prev_lp = lp
            if delta < self.tol:
                converged = True
                break

        return _FitState(
            pi=pi,
            log_posterior=prev_lp,
            log_posterior_history=history,
            n_iter=n_iter,
            converged=converged,
        )

    def _build_prior_alpha_vec(
        self, A: np.ndarray, y: np.ndarray, T: int,
    ) -> np.ndarray:
        """Per-taxon Dirichlet concentration vector ``a_t`` for the M-step.

        ``"symmetric"``      -> ``a_t = alpha`` for every t (legacy behavior).
        ``"empirical_bayes"`` -> ``a_t = prior_kappa * pi_hat_unique[t] + prior_alpha0``,
        where ``pi_hat_unique`` is built from the same PSM-weighted unique
        signal used by ``init='unique_psm'`` (small epsilon + L1 normalise).
        """
        if self.prior_mode == "symmetric":
            return np.full(T, self.alpha, dtype=np.float64)

        raw = self._compute_unique_psm_vector(A, y)
        # Match the epsilon used by the init path so the prior anchor is
        # well-defined when some taxa have no unique PSMs.
        raw = raw + 1e-3
        pi_hat_unique = raw / raw.sum()
        return self.prior_kappa * pi_hat_unique + self.prior_alpha0

    def _em_step(self, pi: np.ndarray, M: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One full EM iteration (E-step + M-step).

        M-step (unified across prior modes):

            pi_t^new = (C_t + a_t - 1) / (N + sum_t (a_t - 1))

        where ``C_t = sum_p y_p r_{pt}`` is the expected count contributed to
        taxon t and ``a_t`` is the per-taxon Dirichlet concentration computed
        once by :meth:`_build_prior_alpha_vec`.  In the symmetric mode every
        ``a_t == alpha`` and the denominator collapses to the legacy
        ``N + T * (alpha - 1)`` form, reproducing the prior behaviour
        bit-for-bit.
        """
        T = pi.shape[0]

        # phi_p = sum_t pi_t * M_{pt}
        phi = M @ pi  # shape (P,)
        phi = np.maximum(phi, _EPS)

        # weight_p = y_p / phi_p; expected counts per (p, t) are y_p * r_{pt}
        # = (pi_t / phi_p) * y_p * M_{pt}. Sum over p:
        # sum_p c_{pt} = pi_t * sum_p (M_{pt} * y_p / phi_p).
        weighted = y / phi  # shape (P,)
        col_sum = M.T @ weighted  # shape (T,)
        expected_counts_t = pi * col_sum  # sum_p c_{pt}

        N = float(y.sum())
        a_vec = self._prior_alpha_vec
        if a_vec is None:  # defensive — fit() always sets this before EM
            a_vec = np.full(T, self.alpha, dtype=np.float64)
        prior_correction = a_vec - 1.0  # per-taxon (a_t - 1)
        denom = N + float(prior_correction.sum())
        if denom <= 0:
            # Pathological combination of small N and very sparse prior.
            denom = max(denom, _EPS)
        numer = expected_counts_t + prior_correction
        pi_new = numer / denom

        # Clamp negative entries that arise when any a_t < 1 and a taxon
        # picks up almost no expected counts. Renormalize so we stay on the
        # simplex.
        pi_new = np.maximum(pi_new, _EPS)
        pi_new = pi_new / pi_new.sum()
        return pi_new

    @staticmethod
    def _responsibilities(pi: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Compute the E-step responsibility matrix r_{pt}."""
        phi = M @ pi  # (P,)
        phi_safe = np.maximum(phi, _EPS)
        # r_{pt} = (pi_t * M_{pt}) / phi_p, broadcast across t.
        r = (M * pi[np.newaxis, :]) / phi_safe[:, np.newaxis]
        return r

    def _log_posterior(self, pi: np.ndarray, M: np.ndarray, y: np.ndarray) -> float:
        """Log-posterior up to a normalization constant.

            log p(pi | y) propto  sum_p y_p log(phi_p)
                                + sum_t (a_t - 1) * log pi_t

        In the symmetric mode every ``a_t == alpha`` and the prior term
        reduces to ``(alpha - 1) * sum_t log pi_t``; in the empirical-Bayes
        mode each taxon contributes its own per-taxon weight.
        """
        phi = M @ pi
        phi = np.maximum(phi, _EPS)
        ll = float(np.sum(y * np.log(phi)))
        # Skip the prior contribution from clamped components — they
        # contribute the same constant under both old and new pi (both at
        # _EPS), so the monotonicity check is preserved.
        pi_safe = np.maximum(pi, _EPS)
        a_vec = self._prior_alpha_vec
        if a_vec is None:
            a_vec = np.full(pi.shape[0], self.alpha, dtype=np.float64)
        prior = float(np.sum((a_vec - 1.0) * np.log(pi_safe)))
        return ll + prior

    def _compute_standard_errors(
        self,
        pi: np.ndarray,
        M: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Approximate standard errors from the observed Fisher information.

        We treat ``pi`` as an unconstrained T-vector for the purposes of this
        rough heuristic and compute

            F_{tt'} = sum_p (M_{pt} * M_{pt'} * y_p) / phi_p^2

        then SE_t = sqrt(diag(F^{-1})_t). If F is rank-deficient, we use
        ``scipy.linalg.pinvh`` and set ``self.fisher_singular_ = True``.
        """
        T = pi.shape[0]
        if T == 1:
            return np.zeros(1)

        phi = M @ pi
        phi_safe = np.maximum(phi, _EPS)
        weights = y / (phi_safe ** 2)  # (P,)
        # F = M^T diag(weights) M  -> shape (T, T)
        F = (M * weights[:, np.newaxis]).T @ M

        # Try a regular inverse first; fall back to pseudoinverse on failure.
        ridge = 1e-10 * np.trace(F) / max(T, 1)
        F_reg = F + ridge * np.eye(T)
        try:
            F_inv = sla.inv(F_reg)
            self.fisher_singular_ = False
        except (sla.LinAlgError, ValueError):
            F_inv = sla.pinvh(F_reg)
            self.fisher_singular_ = True
            logger.warning(
                "Fisher information matrix is singular; using pseudoinverse "
                "for standard errors."
            )

        diag = np.diag(F_inv)
        # Numerical safety: a near-singular F can give tiny negative diagonals.
        diag = np.maximum(diag, 0.0)
        return np.sqrt(diag)

    def _confidences(self) -> np.ndarray:
        """Map standard errors to a [0, 1] confidence heuristic.

        ``confidence_t = max(0, 1 - 2 * SE_t)``. This is a rough proxy: a
        standard error of zero gives confidence 1.0, and a standard error of
        0.5 or more gives confidence 0.0. The factor of 2 corresponds loosely
        to a 95% z-interval half-width covering the full simplex side.
        """
        if self.standard_errors_ is None:
            return np.zeros(self.pi_.shape[0] if self.pi_ is not None else 0)
        return np.clip(1.0 - 2.0 * self.standard_errors_, 0.0, 1.0)
