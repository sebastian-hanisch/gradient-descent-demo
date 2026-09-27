"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt, Konditionszahl-Sweep (gemessene vs.
theoretische Konvergenzrate), Schrittweiten-Divergenzschwelle, Backtracking-vs-fest-Vergleich,
Rosenbrock-Lauf, exakte-Liniensuche-Korrektheits-Kette, Gradienten-Check."""
from dataclasses import dataclass

import numpy as np

import gd_functions as fn
import gd_optimizer as opt


@dataclass(frozen=True)
class Settings:
    dim: int
    condition_number: float
    mode: str
    eta: float
    max_iter: int
    seed: int


def analyse(settings: Settings) -> dict:
    A = fn.random_spd_matrix(settings.dim, settings.condition_number, settings.seed)
    f, grad, hess = fn.quadratic(A)
    rng = np.random.default_rng(settings.seed + 1)
    x0 = rng.normal(size=settings.dim)
    result = opt.gradient_descent(f, grad, x0, mode=settings.mode, eta=settings.eta, A=A,
                                   max_iter=settings.max_iter)
    return {"A": A, "x0": x0, "result": result, "f_star": 0.0}


def analyse_rosenbrock(mode: str, eta: float, max_iter: int, x0=(-1.2, 1.0)) -> dict:
    f, grad, hess = fn.rosenbrock()
    result = opt.gradient_descent(f, grad, np.array(x0, dtype=float), mode=mode, eta=eta,
                                   max_iter=max_iter)
    return {"result": result, "f_star": 0.0, "x_star": np.array([1.0, 1.0])}


def condition_number_sweep(kappas=(2, 10, 50, 200), dim=10, seed=0, max_iter=500) -> list:
    """Gemessene Konvergenzrate (exakte Liniensuche) gegen die klassische Schranke
    ((kappa-1)/(kappa+1))^2 (Nocedal & Wright, Numerical Optimization, Thm. 3.4)."""
    rows = []
    for kappa in kappas:
        A = fn.random_spd_matrix(dim, kappa, seed)
        f, grad, hess = fn.quadratic(A)
        rng = np.random.default_rng(seed + 1)
        x0 = rng.normal(size=dim)
        result = opt.gradient_descent(f, grad, x0, mode="exact", A=A, max_iter=max_iter)
        fvals = result.fvals
        window = min(len(fvals), 50)
        eps = 1e-300
        ratios = fvals[1:window] / np.maximum(fvals[: window - 1], eps)
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        measured_rate = float(np.exp(np.mean(np.log(ratios)))) if len(ratios) else float("nan")
        theoretical_rate = ((kappa - 1) / (kappa + 1)) ** 2
        rows.append({"kappa": kappa, "iters_to_converge": result.n_iter,
                     "measured_rate": measured_rate, "theoretical_rate": theoretical_rate})
    return rows


def step_size_divergence_threshold(dim=6, kappa=20.0, seed=0,
                                    factors=(0.5, 0.9, 0.99, 1.01, 1.1, 1.5)) -> dict:
    """Feste Schrittweite ueber/unter der theoretischen Schwelle eta < 2/lambda_max."""
    A = fn.random_spd_matrix(dim, kappa, seed)
    f, grad, hess = fn.quadratic(A)
    lam_max = float(np.linalg.eigvalsh(A).max())
    theoretical_threshold = 2.0 / lam_max
    rng = np.random.default_rng(seed + 1)
    x0 = rng.normal(size=dim)
    rows = []
    for factor in factors:
        eta = factor * theoretical_threshold
        result = opt.gradient_descent(f, grad, x0, mode="fixed", eta=eta, max_iter=200)
        diverged = (not np.all(np.isfinite(result.fvals))) or result.fvals[-1] > result.fvals[0] * 10
        rows.append({"factor": factor, "eta": eta, "diverged": bool(diverged),
                     "f_final": float(result.fvals[-1]) if np.isfinite(result.fvals[-1]) else float("inf")})
    return {"theoretical_threshold": theoretical_threshold, "rows": rows}


def backtracking_vs_fixed(kappas=(5, 20, 100, 500, 2000), dim=10, seed=0, max_iter=2000) -> list:
    rows = []
    for kappa in kappas:
        A = fn.random_spd_matrix(dim, kappa, seed)
        f, grad, hess = fn.quadratic(A)
        lam_max = float(np.linalg.eigvalsh(A).max())
        rng = np.random.default_rng(seed + 1)
        x0 = rng.normal(size=dim)
        eta_fixed = 1.0 / lam_max  # klassische "sichere" feste Schrittweite
        r_fixed = opt.gradient_descent(f, grad, x0, mode="fixed", eta=eta_fixed, max_iter=max_iter)
        r_bt = opt.gradient_descent(f, grad, x0, mode="backtracking", max_iter=max_iter)
        rows.append({"kappa": kappa, "fixed_iters": r_fixed.n_iter,
                     "fixed_final": float(r_fixed.fvals[-1]), "backtrack_iters": r_bt.n_iter,
                     "backtrack_final": float(r_bt.fvals[-1])})
    return rows


def exact_line_search_check(dim=8, n_trials=5, seed=0) -> float:
    """Maximale relative Abweichung der Code-Schrittweite von der geschlossenen Formel."""
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for i in range(n_trials):
        A = fn.random_spd_matrix(dim, 20.0, seed=seed + i)
        f, grad, hess = fn.quadratic(A)
        x = rng.normal(size=dim)
        g = grad(x)
        eta_formula = opt.exact_step_formula(g, A)
        etas = np.linspace(0.0, 2 * eta_formula, 20001)[1:]
        vals = [f(x - e * g) for e in etas]
        eta_numeric = float(etas[int(np.argmin(vals))])
        err = abs(eta_formula - eta_numeric) / eta_formula
        max_err = max(max_err, err)
    return max_err


def gradient_check(eps: float = 1e-6) -> dict:
    rng = np.random.default_rng(0)
    A = fn.random_spd_matrix(6, 10.0, seed=1)
    f, grad, hess = fn.quadratic(A)
    x = rng.normal(size=6)
    analytic = grad(x)
    numeric = np.zeros(6)
    for i in range(6):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric[i] = (f(xp) - f(xm)) / (2 * eps)
    quad_err = float(np.max(np.abs(analytic - numeric) / np.maximum(np.abs(analytic), 1e-8)))

    f2, grad2, hess2 = fn.rosenbrock()
    x2 = np.array([0.3, -0.7])
    analytic2 = grad2(x2)
    numeric2 = np.zeros(2)
    for i in range(2):
        xp, xm = x2.copy(), x2.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric2[i] = (f2(xp) - f2(xm)) / (2 * eps)
    rosen_err = float(np.max(np.abs(analytic2 - numeric2) / np.maximum(np.abs(analytic2), 1e-8)))
    return {"quadratic_max_rel_err": quad_err, "rosenbrock_max_rel_err": rosen_err}
