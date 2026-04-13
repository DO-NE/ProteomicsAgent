"""Multinomial-mixture EM (MAP) for taxon abundance estimation.

The model treats observed peptide spectral counts as draws from a mixture of
taxon-specific peptide emission distributions. Inference is performed by
Expectation-Maximization on the log-posterior with a sparse Dirichlet prior.

Generative model
----------------
- A in {0, 1}^{P x T} : peptide-to-taxon mapping (1 if taxon t can produce
  peptide p).
- d in R_+^P : per-peptide detectability weights (default all-ones = uniform).
- M_{pt} = (A_{pt} * d_p) / sum_{p'} (A_{p't} * d_{p'}) : weighted emission.
  When d = 1, this reduces to the uniform baseline A_{pt} / n_t.
- pi in simplex^{T-1} : taxon abundance vector (the inference target).
- phi_p(pi) = sum_t pi_t * M_{pt} : marginal probability of peptide p.
- y ~ Multinomial(N, phi(pi)).
- pi ~ Dirichlet(alpha) prior, alpha < 1 encourages sparsity.

EM updates
----------
E-step:
    r_{pt} = (pi_t * M_{pt}) / phi_p
    c_{pt} = y_p * r_{pt}

M-step (MAP with Dirichlet(alpha)):
    pi_t^new = (sum_p c_{pt} + alpha - 1) / (N + T * (alpha - 1))

Components that go non-positive (possible when alpha < 1) are clamped to a
small floor and the vector is renormalized.
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

    Supports optional per-peptide detectability weights that correct the
    emission probabilities for non-uniform peptide observability.  When
    weights are provided, the emission matrix becomes
    ``M_{pt} = (A_{pt} * d_p) / sum_{p'} (A_{p't} * d_{p'})``.

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
    init : {"unique", "uniform", "random"}, optional
        Initialization strategy for the first run (default ``"unique"``).
        ``"unique"`` weights each taxon by the count of unique peptides that
        also have nonzero spectral counts; ``"uniform"`` uses 1/T; ``"random"``
        samples from Dirichlet(1, ..., 1). Restarts always use random.
    seed : int or None, optional
        Random seed for reproducibility.

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
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if tol <= 0:
            raise ValueError("tol must be > 0")
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        if init not in ("unique", "uniform", "random"):
            raise ValueError("init must be 'unique', 'uniform', or 'random'")

        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_restarts = int(n_restarts)
        self.min_abundance = float(min_abundance)
        self.init = init
        self.seed = seed

        # Set after fit().
        self.pi_: Optional[np.ndarray] = None
        self.responsibilities_: Optional[np.ndarray] = None
        self.log_posterior_history_: list = []
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.standard_errors_: Optional[np.ndarray] = None
        self.fisher_singular_: bool = False

        # Cached internals (not part of public API).
        self._A: Optional[np.ndarray] = None
        self._M: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._taxon_names: Optional[list] = None
        self._detectability_weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ public

    def fit(
        self,
        A: np.ndarray,
        y: np.ndarray,
        detectability_weights: Optional[np.ndarray] = None,
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
        detectability_weights : np.ndarray or None, shape ``(P,)``
            Per-peptide detectability weights.  When provided, the emission
            matrix is computed as
            ``M_{pt} = (A_{pt} * d_p) / sum_{p'} (A_{p't} * d_{p'})``
            instead of the uniform ``A_{pt} / n_t``.  Pass ``None`` (default)
            for the original uniform-emission model.

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

        # Validate detectability weights if provided.
        if detectability_weights is not None:
            detectability_weights = np.asarray(
                detectability_weights, dtype=np.float64
            )
            if detectability_weights.shape != (P,):
                raise ValueError(
                    f"detectability_weights must have shape ({P},), "
                    f"got {detectability_weights.shape}"
                )
            if (detectability_weights < 0).any():
                raise ValueError("detectability_weights must be non-negative")

        # Coerce to binary float matrix once.
        A_bin = (A_arr != 0).astype(np.float64)

        # Build emission matrix M, optionally weighted by detectability.
        from .detectability import build_weighted_emission_matrix

        M = build_weighted_emission_matrix(A_bin, detectability_weights)

        # Edge case: T == 1 forces pi = [1.0]; skip EM entirely.
        if T == 1:
            pi = np.array([1.0])
            responsibilities = np.zeros((P, 1), dtype=np.float64)
            mask = y_arr > 0
            responsibilities[mask, 0] = 1.0
            self._A = A_bin
            self._M = M
            self._y = y_arr
            self._detectability_weights = detectability_weights
            self.pi_ = pi
            self.responsibilities_ = responsibilities
            self.log_posterior_history_ = [self._log_posterior(pi, M, y_arr)]
            self.converged_ = True
            self.n_iter_ = 0
            self.standard_errors_ = self._compute_standard_errors(pi, M, y_arr)
            return self

        rng = np.random.default_rng(self.seed)

        best: Optional[_FitState] = None
        for restart in range(self.n_restarts):
            if restart == 0:
                pi0 = self._initial_pi(self.init, T, A_bin, y_arr, rng)
            else:
                pi0 = self._initial_pi("random", T, A_bin, y_arr, rng)
            state = self._run_em(pi0, M, y_arr)
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

        responsibilities = self._responsibilities(pi, M)

        self._A = A_bin
        self._M = M
        self._y = y_arr
        self._detectability_weights = detectability_weights
        self.pi_ = pi
        self.responsibilities_ = responsibilities
        self.log_posterior_history_ = best.log_posterior_history
        self.converged_ = best.converged
        self.n_iter_ = best.n_iter
        self.standard_errors_ = self._compute_standard_errors(pi, M, y_arr)
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

    def _em_step(self, pi: np.ndarray, M: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One full EM iteration (E-step + M-step)."""
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
        prior_correction = self.alpha - 1.0
        denom = N + T * prior_correction
        if denom <= 0:
            # Pathological combination of small N and very sparse prior.
            denom = max(denom, _EPS)
        numer = expected_counts_t + prior_correction
        pi_new = numer / denom

        # Clamp negative entries that arise when alpha < 1 and a taxon picks
        # up almost no expected counts. Renormalize so we stay on the simplex.
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

        log p(pi | y) propto sum_p y_p log(phi_p) + (alpha - 1) sum_t log pi_t.
        """
        phi = M @ pi
        phi = np.maximum(phi, _EPS)
        ll = float(np.sum(y * np.log(phi)))
        # Skip the prior contribution from clamped components — they
        # contribute the same constant under both old and new pi (both at
        # _EPS), so the monotonicity check is preserved.
        pi_safe = np.maximum(pi, _EPS)
        prior = float((self.alpha - 1.0) * np.sum(np.log(pi_safe)))
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
