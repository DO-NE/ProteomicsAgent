"""Unit tests for the Cycle-7 prior / initialiser knobs on ``AbundanceEM``.

Covered surfaces
----------------
* ``init='unique_psm'`` — PSM-weighted unique-peptide initialiser.
* ``prior_mode='empirical_bayes'`` — asymmetric Dirichlet anchored on the
  PSM-weighted unique vector (``a_t = prior_kappa * pi_hat_unique + prior_alpha0``).
* Backwards-compatibility: ``empirical_bayes`` with ``prior_kappa=0`` and
  ``prior_alpha0=alpha`` must collapse exactly to the legacy symmetric path.
* Log-posterior monotonicity carries over to the EB prior.

Run from the repository root with::

    python -m pytest taxon/algorithms/abundance_em_core/tests/test_prior_and_init.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from taxon.algorithms.abundance_em_core.model import AbundanceEM
from taxon.algorithms.abundance_em_core.synthetic import (
    generate_synthetic_community,
)


def _toy_two_taxon_case():
    """Build the (P=7, T=2) toy matrix used by the ``unique_psm`` test.

    * Taxon 0 has 2 unique peptides with ``y = [10, 10]`` (sum = 20).
    * Taxon 1 has 5 unique peptides with ``y = [1, 1, 1, 1, 1]`` (sum = 5).

    Under ``unique`` initialisation (count-based) taxon 1 dominates because
    it has more unique rows; under ``unique_psm`` (PSM-weighted) taxon 0
    dominates because its unique peptides carry far more spectra.  The two
    modes must disagree here — that is the whole point of the new init.
    """
    A = np.zeros((7, 2), dtype=np.int8)
    A[0, 0] = 1
    A[1, 0] = 1
    A[2, 1] = 1
    A[3, 1] = 1
    A[4, 1] = 1
    A[5, 1] = 1
    A[6, 1] = 1
    y = np.array([10, 10, 1, 1, 1, 1, 1], dtype=np.float64)
    return A, y


# --------------------------------------------------------------------- Feature 1


def test_init_unique_psm():
    """``unique_psm`` weights by PSM mass; ``unique`` weights by peptide count."""
    A, y = _toy_two_taxon_case()
    rng = np.random.default_rng(0)
    T = A.shape[1]

    pi0_unique = AbundanceEM._initial_pi("unique", T, A, y, rng)
    pi0_psm = AbundanceEM._initial_pi("unique_psm", T, A, y, rng)

    # Count-based: 2 vs 5 unique peptides → ~[2/7, 5/7] after the 1e-3 eps.
    expected_unique = np.array([2.001, 5.001])
    expected_unique /= expected_unique.sum()
    np.testing.assert_allclose(pi0_unique, expected_unique, atol=1e-9)

    # PSM-weighted: 20 vs 5 PSMs on unique rows → ~[20/25, 5/25] after eps.
    expected_psm = np.array([20.001, 5.001])
    expected_psm /= expected_psm.sum()
    np.testing.assert_allclose(pi0_psm, expected_psm, atol=1e-9)

    # The two modes MUST disagree on this case — that is what makes the new
    # initialiser useful (sanity guard against a silent revert).
    assert not np.allclose(pi0_unique, pi0_psm, atol=1e-3), (
        f"unique_psm and unique produced indistinguishable pi_0: "
        f"{pi0_unique} vs {pi0_psm}"
    )

    # Spot-check the leading entry against the hand-computed fraction so
    # the test fails loudly if the formula ever drifts.
    assert pi0_psm[0] == pytest.approx(20.001 / 25.002, abs=1e-9)
    assert pi0_unique[0] == pytest.approx(2.001 / 7.002, abs=1e-9)


def test_init_unique_psm_constructor_accepts():
    """The model constructor must accept the new init enum value."""
    # Should not raise.
    AbundanceEM(init="unique_psm")
    # Bad value still raises.
    with pytest.raises(ValueError):
        AbundanceEM(init="not_a_mode")


# --------------------------------------------------------------------- Feature 2


def test_prior_backward_compat():
    """EB with ``prior_kappa=0`` and ``prior_alpha0=alpha`` == symmetric run.

    With ``prior_kappa=0`` the per-taxon Dirichlet vector collapses to a
    constant ``prior_alpha0`` for every taxon; matching it to ``alpha``
    therefore reproduces the symmetric M-step bit-for-bit (within fp noise).
    """
    data = generate_synthetic_community(
        n_taxa=5,
        n_peptides_per_taxon=120,
        shared_fraction=0.20,
        total_psms=6000,
        seed=7,
    )

    alpha = 0.7
    sym = AbundanceEM(
        alpha=alpha, max_iter=500, tol=1e-9, seed=0,
        prior_mode="symmetric",
    )
    sym.fit(data["A"], data["y"])

    eb = AbundanceEM(
        alpha=alpha, max_iter=500, tol=1e-9, seed=0,
        prior_mode="empirical_bayes",
        prior_kappa=0.0,
        prior_alpha0=alpha,
    )
    eb.fit(data["A"], data["y"])

    np.testing.assert_allclose(eb.pi_, sym.pi_, atol=1e-10)
    assert eb.n_iter_ == sym.n_iter_


def test_prior_empirical_bayes_pulls_toward_unique():
    """Increasing ``prior_kappa`` monotonically pulls pi_ toward pi_hat_unique.

    Construction: 3 taxa, where taxon 2 has *no* unique peptides but shares
    a large block with taxa 0 and 1.  The MLE distributes the shared mass
    across all three taxa, so taxon 2 picks up sizeable π even though its
    unique-PSM evidence is zero.  pi_hat_unique therefore has near-zero mass
    on taxon 2, and a stronger EB prior must drag π_2 down.
    """
    rng = np.random.default_rng(0)
    P_unique_0, P_unique_1 = 30, 30
    P_shared = 60
    P = P_unique_0 + P_unique_1 + P_shared
    T = 3
    A = np.zeros((P, T), dtype=np.int8)
    A[:P_unique_0, 0] = 1
    A[P_unique_0:P_unique_0 + P_unique_1, 1] = 1
    # Shared block: maps to taxa 0, 1, AND 2 (so the MLE can split it across
    # three taxa even though taxon 2 has no unique peptides of its own).
    A[P_unique_0 + P_unique_1:, 0] = 1
    A[P_unique_0 + P_unique_1:, 1] = 1
    A[P_unique_0 + P_unique_1:, 2] = 1

    y = np.zeros(P, dtype=np.float64)
    y[:P_unique_0] = 30.0   # heavy unique evidence for taxon 0
    y[P_unique_0:P_unique_0 + P_unique_1] = 5.0  # light unique for taxon 1
    y[P_unique_0 + P_unique_1:] = 20.0  # plenty of shared spectra

    # PSM-weighted unique vector that the prior anchors on.
    raw = AbundanceEM._compute_unique_psm_vector(A, y) + 1e-3
    pi_hat_unique = raw / raw.sum()

    def fit_with_kappa(kappa: float) -> np.ndarray:
        m = AbundanceEM(
            alpha=1.0, max_iter=2000, tol=1e-10, seed=0,
            min_abundance=0.0,           # don't post-threshold tiny mass
            prior_mode="empirical_bayes",
            prior_kappa=kappa,
            prior_alpha0=1.0,            # neutral baseline → mass entirely
                                         # from N * π_MLE + kappa * π_hat
        )
        m.fit(A, y)
        return m.pi_

    pis = [fit_with_kappa(k) for k in (0.0, 1e3, 1e5)]
    distances = [float(np.abs(p - pi_hat_unique).sum()) for p in pis]

    # As kappa grows the EM solution must move at least as close to
    # pi_hat_unique (monotonic non-increasing L1 distance).
    assert distances[1] < distances[0], (
        f"raising kappa from 0 to 1e3 did not pull π toward pi_hat_unique: "
        f"distances={distances}"
    )
    assert distances[2] < distances[1], (
        f"raising kappa from 1e3 to 1e5 did not pull π toward pi_hat_unique: "
        f"distances={distances}"
    )
    # Very large kappa should drive π essentially to pi_hat_unique.
    assert distances[2] < 5e-2, (
        f"large kappa failed to collapse π onto pi_hat_unique: distance={distances[2]}"
    )


def test_yaml_config_wiring():
    """``em.prior_mode / prior_kappa / prior_alpha0`` survive the YAML pipeline.

    Guards against the regression that motivated the wiring work: dropping
    the new knobs into ``em:`` in a YAML file must reach the plugin config
    (via ``_flatten_yaml_config``), project to env vars
    (via ``_apply_config_to_env``), and round-trip back into ``em:`` when
    ``_serialize_run_config`` writes ``run_config.yaml``.
    """
    import importlib
    import os
    import sys
    import tempfile
    from pathlib import Path

    # ``main`` lives at the repo root, not on the package path; locate it
    # relative to this test file so the test stays robust to the cwd.
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    main_mod = importlib.import_module("main")

    yaml_blob = """
input: /tmp/foo.mzML
db: /tmp/foo.fasta
em:
  alpha: 0.5
  init_strategy: unique_psm
  prior_mode: empirical_bayes
  prior_kappa: 50.0
  prior_alpha0: 0.5
"""
    import yaml as _yaml
    raw = _yaml.safe_load(yaml_blob)

    flat = main_mod._flatten_yaml_config(raw)
    assert flat["init_strategy"] == "unique_psm"
    assert flat["prior_mode"] == "empirical_bayes"
    assert flat["prior_kappa"] == pytest.approx(50.0)
    assert flat["prior_alpha0"] == pytest.approx(0.5)

    # Project to env and check the orchestrator-visible variables.
    saved = {k: os.environ.get(k) for k in (
        "TAXON_EM_PRIOR_MODE", "TAXON_EM_PRIOR_KAPPA",
        "TAXON_EM_PRIOR_ALPHA0", "TAXON_EM_INIT",
    )}
    try:
        for k in saved:
            os.environ.pop(k, None)
        main_mod._apply_config_to_env(flat)
        assert os.environ["TAXON_EM_PRIOR_MODE"] == "empirical_bayes"
        assert os.environ["TAXON_EM_PRIOR_KAPPA"] == "50.0"
        assert os.environ["TAXON_EM_PRIOR_ALPHA0"] == "0.5"
        assert os.environ["TAXON_EM_INIT"] == "unique_psm"
    finally:
        # Restore prior env state so other tests are not perturbed.
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # Round-trip back to YAML — the three knobs must live under ``em:``.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        out_path = Path(f.name)
    try:
        main_mod._serialize_run_config(flat, out_path)
        out_raw = _yaml.safe_load(out_path.read_text())
        assert out_raw["em"]["prior_mode"] == "empirical_bayes"
        assert out_raw["em"]["prior_kappa"] == pytest.approx(50.0)
        assert out_raw["em"]["prior_alpha0"] == pytest.approx(0.5)
        assert out_raw["em"]["init_strategy"] == "unique_psm"
        # Defensive: nothing should have leaked into the catch-all ``_extra``.
        assert "_extra" not in out_raw or not any(
            k in out_raw.get("_extra", {})
            for k in ("prior_mode", "prior_kappa", "prior_alpha0")
        )
    finally:
        out_path.unlink(missing_ok=True)


def test_log_posterior_monotonic_eb():
    """``log_posterior_history_`` is non-decreasing under EB with kappa > 0."""
    data = generate_synthetic_community(
        n_taxa=6,
        n_peptides_per_taxon=150,
        shared_fraction=0.20,
        total_psms=8000,
        seed=5,
    )

    model = AbundanceEM(
        alpha=0.7, max_iter=300, tol=1e-9, seed=0,
        prior_mode="empirical_bayes",
        prior_kappa=50.0,
        prior_alpha0=0.5,
    )
    model.fit(data["A"], data["y"])

    history = np.asarray(model.log_posterior_history_, dtype=np.float64)
    assert history.size >= 2
    diffs = np.diff(history)
    # Same tolerance scheme as ``test_em_monotonicity`` — 1e-7 relative to
    # |lp|, which absorbs ULP-level fp noise without hiding real regressions.
    assert (diffs >= -1e-7 * (np.abs(history[:-1]) + 1.0)).all(), (
        f"non-monotonic step under EB: {diffs.min()}"
    )
