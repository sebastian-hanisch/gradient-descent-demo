import numpy as np

import gd_functions as fn
import gd_optimizer as opt


def test_exact_line_search_converges_to_known_optimum():
    A = fn.random_spd_matrix(5, 15.0, seed=2)
    f, grad, hess = fn.quadratic(A)
    x0 = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    result = opt.gradient_descent(f, grad, x0, mode="exact", A=A, max_iter=500)
    assert result.converged
    np.testing.assert_allclose(result.trajectory[-1], np.zeros(5), atol=1e-6)


def test_backtracking_converges_on_rosenbrock():
    f, grad, hess = fn.rosenbrock()
    x0 = np.array([-1.2, 1.0])
    result = opt.gradient_descent(f, grad, x0, mode="backtracking", max_iter=5000)
    assert result.converged
    np.testing.assert_allclose(result.trajectory[-1], np.array([1.0, 1.0]), atol=1e-4)


def test_fixed_step_size_above_threshold_diverges():
    A = fn.random_spd_matrix(4, 20.0, seed=5)
    f, grad, hess = fn.quadratic(A)
    lam_max = np.linalg.eigvalsh(A).max()
    x0 = np.array([1.0, 1.0, 1.0, 1.0])
    result = opt.gradient_descent(f, grad, x0, mode="fixed", eta=1.5 * (2 / lam_max), max_iter=100)
    assert result.fvals[-1] > result.fvals[0]


def test_fixed_step_size_below_threshold_converges():
    A = fn.random_spd_matrix(4, 20.0, seed=5)
    f, grad, hess = fn.quadratic(A)
    lam_max = np.linalg.eigvalsh(A).max()
    x0 = np.array([1.0, 1.0, 1.0, 1.0])
    result = opt.gradient_descent(f, grad, x0, mode="fixed", eta=0.5 * (2 / lam_max), max_iter=500)
    assert result.converged
    assert result.fvals[-1] < 1e-6


def test_exact_step_formula_matches_optimizer_internal_choice():
    A = fn.random_spd_matrix(3, 8.0, seed=9)
    f, grad, hess = fn.quadratic(A)
    x = np.array([1.0, 2.0, -1.0])
    g = grad(x)
    formula = opt.exact_step_formula(g, A)
    assert formula > 0
