"""Korrektheits-/Verhaltenstests mit billigen Parametern. Die offiziellen, teuren Sweep-Werte
stehen mit Toleranzband in test_claims.py (Modul-Fixtures, je Sweep nur einmal berechnet)."""
import gd_evaluation as ev


def test_condition_number_sweep_runs_with_small_values():
    rows = ev.condition_number_sweep(kappas=(2, 20), dim=4, max_iter=50)
    assert len(rows) == 2
    for row in rows:
        assert row["theoretical_rate"] < 1.0
        assert 0.0 <= row["measured_rate"] <= 1.0


def test_step_size_divergence_threshold_is_positive():
    out = ev.step_size_divergence_threshold(dim=3)
    assert out["theoretical_threshold"] > 0


def test_backtracking_vs_fixed_backtracking_never_needs_more_iterations():
    rows = ev.backtracking_vs_fixed(kappas=(10, 100), dim=4, max_iter=500)
    for row in rows:
        assert row["backtrack_iters"] <= row["fixed_iters"]


def test_exact_line_search_check_is_near_machine_precision():
    assert ev.exact_line_search_check(n_trials=2) < 1e-6


def test_analyse_returns_valid_result():
    settings = ev.Settings(dim=3, condition_number=10.0, mode="backtracking", eta=0.1,
                           max_iter=200, seed=0)
    out = ev.analyse(settings)
    assert out["result"].fvals[-1] >= 0.0
