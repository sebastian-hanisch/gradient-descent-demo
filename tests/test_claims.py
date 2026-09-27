"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen neu berechnet.
Modul-Fixtures berechnen jeden Sweep nur einmal; Toleranzband statt exakter Gleichheit von
Anfang an (Hauslehre aus rnn-demo/lstm-demo: chaotische Trajektorien ueber viele Schritte sind
nicht bit-reproduzierbar zwischen Plattformen). Hier sind die meisten Groessen aber deterministische
lineare Algebra (fester Seed, keine Trainings-Chaotik) und daher eng tolerierbar."""
import pytest

import gd_evaluation as ev


@pytest.fixture(scope="module")
def condition_sweep_rows():
    return ev.condition_number_sweep()


@pytest.fixture(scope="module")
def backtracking_rows():
    return ev.backtracking_vs_fixed()


@pytest.fixture(scope="module")
def divergence_data():
    return ev.step_size_divergence_threshold()


def test_claim_measured_rate_stays_below_theoretical_bound(condition_sweep_rows):
    """Die klassische Schranke ((k-1)/(k+1))^2 haelt bei jeder geprueften Konditionszahl -
    wie die Novikoff-Schranke im Perceptron-Stueck: ein gueltiges, aber grobes
    Worst-Case-Versprechen."""
    for row in condition_sweep_rows:
        assert row["measured_rate"] <= row["theoretical_rate"] + 1e-6


def test_claim_convergence_rate_worsens_with_condition_number(condition_sweep_rows):
    rates = [r["measured_rate"] for r in condition_sweep_rows]
    assert rates == sorted(rates)


def test_claim_exact_line_search_matches_closed_form_formula():
    assert ev.exact_line_search_check() < 1e-8


def test_claim_fixed_step_below_threshold_never_diverges(divergence_data):
    for row in divergence_data["rows"]:
        if row["eta"] < divergence_data["theoretical_threshold"]:
            assert not row["diverged"]


def test_claim_fixed_step_above_threshold_always_diverges(divergence_data):
    for row in divergence_data["rows"]:
        if row["eta"] > divergence_data["theoretical_threshold"]:
            assert row["diverged"]


def test_claim_backtracking_never_needs_more_iterations_than_fixed(backtracking_rows):
    for row in backtracking_rows:
        assert row["backtrack_iters"] <= row["fixed_iters"]


def test_claim_backtracking_solves_high_condition_case_fixed_does_not(backtracking_rows):
    """Bei kappa=2000 erreicht die 'sichere' feste Schrittweite das Budget von 2000 Iterationen,
    ohne wirklich konvergiert zu sein (f_final >> 0); Backtracking konvergiert innerhalb des
    Budgets auf Maschinengenauigkeit."""
    row = next(r for r in backtracking_rows if r["kappa"] == 2000)
    assert row["fixed_final"] > 1e-4
    assert row["backtrack_final"] < 1e-10


def test_claim_gradient_check_below_1e_minus_6():
    out = ev.gradient_check()
    assert out["quadratic_max_rel_err"] < 1e-6
    assert out["rosenbrock_max_rel_err"] < 1e-6


def test_claim_rosenbrock_fixed_step_does_not_fully_converge_in_budget():
    out = ev.analyse_rosenbrock(mode="fixed", eta=0.001, max_iter=5000)
    assert not out["result"].converged
    assert out["result"].fvals[-1] > 1e-4


def test_claim_rosenbrock_backtracking_converges_well_within_budget():
    out = ev.analyse_rosenbrock(mode="backtracking", eta=0.001, max_iter=2000)
    assert out["result"].converged
    assert out["result"].n_iter < 1000
