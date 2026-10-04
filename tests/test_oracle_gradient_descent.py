"""Unabhaengige Orakel-Tests (anderer Rechenweg als der Code):
- Feste Schrittweite auf der Quadratik: geschlossene Formel x_k = V (I - eta*Lambda)^k V^T x_0
  (Eigenzerlegung) liefert Pfad UND die Zahl der Schritte bis ||grad|| < tol.
- Exakte Liniensuche: scipy.optimize.minimize_scalar auf der 1D-Funktion, Funktionswert nach dem
  Schritt aus geschlossener Formel, Kantorowitsch-Schranke ((k-1)/(k+1))^2 je Schritt.
- Rosenbrock/Gradient/Hesse-Matrix gegen scipy.optimize.rosen*.
- Konditionszahl der Testmatrix per eigvalsh."""
import numpy as np
import pytest

import gd_functions as fn
import gd_optimizer as opt

scipy_opt = pytest.importorskip("scipy.optimize")


def _instances(n, seed=11):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        d = int(rng.integers(2, 7))
        kappa = float(np.exp(rng.uniform(np.log(1.5), np.log(100))))
        A = fn.random_spd_matrix(d, kappa, int(rng.integers(0, 10**6)))
        yield A, kappa, rng.normal(size=d)


def test_spd_matrix_has_prescribed_spectrum():
    for A, kappa, _ in _instances(60):
        w = np.linalg.eigvalsh(A)
        assert w[0] == pytest.approx(1.0, rel=1e-8)
        assert w[-1] == pytest.approx(kappa, rel=1e-8)


def test_fixed_step_path_and_step_count_match_closed_form():
    tol = 1e-10
    for A, kappa, x0 in _instances(60):
        w, V = np.linalg.eigh(A)
        eta = 1.0 / w[-1]
        f, g, _ = fn.quadratic(A)
        r = opt.gradient_descent(f, g, x0, mode="fixed", eta=eta, max_iter=3000, tol=tol)
        c0 = V.T @ x0
        # Orakel: erster Index k mit ||A x_k|| < tol, x_k = V diag((1-eta*w)^k) c0
        k_star = next(k for k in range(100000)
                      if np.linalg.norm(w * ((1 - eta * w) ** k * c0)) < tol)
        assert r.converged
        assert r.n_iter == k_star == len(r.trajectory) - 1
        for k in (0, k_star // 2, k_star):
            np.testing.assert_allclose(r.trajectory[k], V @ ((1 - eta * w) ** k * c0),
                                       rtol=1e-6, atol=1e-12)


def test_exact_line_search_vs_scipy_and_kantorovich():
    for A, kappa, x0 in _instances(60):
        f, g, _ = fn.quadratic(A)
        w = np.linalg.eigvalsh(A)
        r = opt.gradient_descent(f, g, x0, mode="exact", A=A, max_iter=15)
        bound = ((w[-1] / w[0] - 1) / (w[-1] / w[0] + 1)) ** 2
        for j in range(len(r.trajectory) - 1):
            x, gg = r.trajectory[j], A @ r.trajectory[j]
            if f(x) < 1e-14:
                break
            t = scipy_opt.minimize_scalar(lambda s: f(x - s * gg), bounds=(0, 2 / w[0]),
                                          method="bounded", options=dict(xatol=1e-13)).x
            used = (x - r.trajectory[j + 1]) @ gg / (gg @ gg)
            assert used == pytest.approx(t, rel=1e-4)
            f_next = f(x) - 0.5 * (gg @ gg) ** 2 / (gg @ A @ gg)
            assert r.fvals[j + 1] == pytest.approx(f_next, rel=1e-6, abs=1e-14)
            assert r.fvals[j + 1] <= bound * r.fvals[j] * (1 + 1e-9)


def test_fixed_step_divergence_matches_spectral_radius():
    for A, kappa, x0 in _instances(60):
        w = np.linalg.eigvalsh(A)
        f, g, _ = fn.quadratic(A)
        for factor in (0.5, 1.5):
            eta = factor * 2 / w[-1]
            r = opt.gradient_descent(f, g, x0, mode="fixed", eta=eta, max_iter=300)
            rho = np.max(np.abs(1 - eta * w))
            assert (rho > 1) == (r.fvals[-1] > r.fvals[0])


def test_rosenbrock_matches_scipy_reference():
    rng = np.random.default_rng(3)
    f, g, h = fn.rosenbrock()
    for _ in range(100):
        x = rng.normal(size=2) * 2
        assert f(x) == pytest.approx(scipy_opt.rosen(x))
        np.testing.assert_allclose(g(x), scipy_opt.rosen_der(x))
        np.testing.assert_allclose(h(x), scipy_opt.rosen_hess(x))


def test_backtracking_reaches_scipy_minimum_on_rosenbrock():
    f, g, _ = fn.rosenbrock()
    x0 = np.array([-1.2, 1.0])
    r = opt.gradient_descent(f, g, x0, mode="backtracking", max_iter=5000)
    sol = scipy_opt.minimize(scipy_opt.rosen, x0, jac=scipy_opt.rosen_der, method="BFGS",
                             options=dict(gtol=1e-10))
    assert r.converged
    np.testing.assert_allclose(r.trajectory[-1], sol.x, atol=1e-4)
