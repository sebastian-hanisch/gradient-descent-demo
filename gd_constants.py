"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

DEFAULT_SEED = 0

FUNC_QUADRATIC = "quadratik"
FUNC_ROSENBROCK = "rosenbrock"
FUNCTIONS = (FUNC_QUADRATIC, FUNC_ROSENBROCK)

MODE_FIXED = "fixed"
MODE_BACKTRACKING = "backtracking"
MODE_EXACT = "exact"
MODES = (MODE_FIXED, MODE_BACKTRACKING, MODE_EXACT)

DIM = 2  # feste Dimension fuer die interaktive Quadratik (2D-Kontur/Trajektorie sichtbar)

KAPPA_MIN, KAPPA_MAX, KAPPA_DEFAULT = 1.5, 500.0, 20.0
ETA_QUADRATIC_MIN, ETA_QUADRATIC_MAX, ETA_QUADRATIC_DEFAULT = 0.001, 1.5, 0.1
ETA_ROSENBROCK_MIN, ETA_ROSENBROCK_MAX, ETA_ROSENBROCK_DEFAULT = 0.0001, 0.01, 0.001
MAX_ITER_MIN, MAX_ITER_MAX, MAX_ITER_DEFAULT = 10, 20000, 200

# Konditionszahl-Sweep (Kern-Hook, teuer genug fuer eine feste Messreihe statt Live-Regler)
SWEEP_KAPPAS = (2, 10, 50, 200)
SWEEP_DIM = 10

# Backtracking-vs-fest-Sweep
BT_KAPPAS = (5, 20, 100, 500, 2000)
BT_DIM = 10

PRESETS = {
    "gut_konditioniert": dict(
        label="Gut konditioniert — schnelle Konvergenz",
        func=FUNC_QUADRATIC, mode=MODE_EXACT, kappa=5.0, eta=ETA_QUADRATIC_DEFAULT,
        max_iter=100, seed=0,
        help="Bei kleiner Konditionszahl (κ=5) konvergiert Gradientenabstieg mit exakter "
             "Liniensuche in wenigen Schritten direkt zum Optimum.",
    ),
    "schlecht_konditioniert": dict(
        label="Schlecht konditioniert — Zickzack",
        func=FUNC_QUADRATIC, mode=MODE_FIXED, kappa=200.0, eta=0.007,
        max_iter=200, seed=42,
        help="Bei großer Konditionszahl (κ=200) mit fester Schrittweite zickzackt der Pfad "
             "quer zum langgestreckten Tal und kommt nur langsam voran.",
    ),
    "rosenbrock_fest": dict(
        label="Rosenbrock — feste Schrittweite (langsam)",
        func=FUNC_ROSENBROCK, mode=MODE_FIXED, kappa=KAPPA_DEFAULT, eta=0.001,
        max_iter=5000, seed=0,
        help="Auf der gekrümmten Rosenbrock-Funktion braucht eine feste, vorsichtige "
             "Schrittweite tausende Schritte und ist selbst dann noch nicht ganz am Ziel.",
    ),
    "rosenbrock_backtracking": dict(
        label="Rosenbrock — Backtracking (schnell)",
        func=FUNC_ROSENBROCK, mode=MODE_BACKTRACKING, kappa=KAPPA_DEFAULT,
        eta=ETA_ROSENBROCK_DEFAULT, max_iter=2000, seed=0,
        help="Dieselbe Aufgabe, aber mit Armijo-Backtracking: die Schrittweite passt sich "
             "selbst an und erreicht das Optimum in einem Bruchteil der Schritte.",
    ),
}
