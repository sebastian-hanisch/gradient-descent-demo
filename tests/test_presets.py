import gd_constants as C
import gd_evaluation as ev
import gd_functions as fn
import gd_optimizer as opt
import numpy as np


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert preset["func"] in C.FUNCTIONS
        assert preset["mode"] in C.MODES


def test_preset_gut_konditioniert_converges_fast():
    p = C.PRESETS["gut_konditioniert"]
    A = fn.random_spd_matrix(C.DIM, p["kappa"], p["seed"])
    f, grad, hess = fn.quadratic(A)
    rng = np.random.default_rng(p["seed"] + 1)
    x0 = rng.normal(size=C.DIM)
    result = opt.gradient_descent(f, grad, x0, mode=p["mode"], eta=p["eta"], A=A,
                                   max_iter=p["max_iter"])
    assert result.converged
    assert result.n_iter < 100


def test_preset_schlecht_konditioniert_is_slow_but_moving():
    p = C.PRESETS["schlecht_konditioniert"]
    A = fn.random_spd_matrix(C.DIM, p["kappa"], p["seed"])
    f, grad, hess = fn.quadratic(A)
    rng = np.random.default_rng(p["seed"] + 1)
    x0 = rng.normal(size=C.DIM)
    result = opt.gradient_descent(f, grad, x0, mode=p["mode"], eta=p["eta"], A=A,
                                   max_iter=p["max_iter"])
    assert result.fvals[-1] < result.fvals[0]  # macht Fortschritt
    assert result.fvals[-1] > 1e-6  # aber noch nicht am Ziel


def test_preset_rosenbrock_backtracking_beats_fest():
    p_fest = C.PRESETS["rosenbrock_fest"]
    p_bt = C.PRESETS["rosenbrock_backtracking"]
    out_fest = ev.analyse_rosenbrock(mode=p_fest["mode"], eta=p_fest["eta"],
                                     max_iter=p_fest["max_iter"])
    out_bt = ev.analyse_rosenbrock(mode=p_bt["mode"], eta=p_bt["eta"], max_iter=p_bt["max_iter"])
    assert out_bt["result"].n_iter < out_fest["result"].n_iter
