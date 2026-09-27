"""Gradientenabstieg mit drei Schrittweiten-Modi: fest, exakte Liniensuche (nur fuer Quadratiken,
geschlossene Formel) und Backtracking/Armijo (allgemein, robust)."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    trajectory: np.ndarray  # (n_steps+1, dim)
    fvals: np.ndarray  # (n_steps+1,)
    converged: bool
    n_iter: int


def gradient_descent(f, grad, x0, mode="backtracking", eta=0.1, A=None, max_iter=1000,
                      tol=1e-10, alpha=0.3, beta=0.5) -> Result:
    """mode: 'fixed' (feste Schrittweite eta), 'exact' (geschlossene Formel, braucht A fuer eine
    Quadratik f(x)=0.5 x^T A x), 'backtracking' (Armijo-Bedingung, Parameter alpha/beta)."""
    x = np.asarray(x0, dtype=float).copy()
    traj = [x.copy()]
    fvals = [float(f(x))]
    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        g = grad(x)
        gnorm2 = float(g @ g)
        if gnorm2 < tol ** 2:
            converged = True
            break
        if mode == "fixed":
            step = eta
        elif mode == "exact":
            if A is None:
                raise ValueError("mode='exact' braucht A (nur fuer Quadratiken definiert)")
            denom = float(g @ A @ g)
            step = gnorm2 / denom if denom > 0 else eta
        elif mode == "backtracking":
            step = 1.0
            fx = fvals[-1]
            while f(x - step * g) > fx - alpha * step * gnorm2:
                step *= beta
                if step < 1e-16:
                    break
        else:
            raise ValueError(f"unbekannter Modus: {mode}")
        x = x - step * g
        traj.append(x.copy())
        fvals.append(float(f(x)))
        if not np.all(np.isfinite(x)):
            break
    else:
        k = max_iter
    return Result(trajectory=np.array(traj), fvals=np.array(fvals), converged=converged, n_iter=k)


def exact_step_formula(g: np.ndarray, A: np.ndarray) -> float:
    """Geschlossene Formel fuer die exakte Liniensuche auf f(x)=0.5 x^T A x."""
    return float(g @ g) / float(g @ A @ g)
