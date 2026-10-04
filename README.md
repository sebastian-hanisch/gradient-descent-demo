# Gradientenabstieg – der erste Schritt bergab – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-gradient-descent-demo.streamlit.app/)**

Erstes Stück (Wurzel) der **Nichtlineare-Optimierung-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Um eine glatte Funktion zu minimieren, ohne mehr über sie zu wissen als ihren Gradienten: gehe
immer entlang des **negativen Gradienten** (Cauchy 1847) – bergab, so lange bis sich nichts mehr
bessert. Wie schnell das geht, hängt stark davon ab, wie **schlecht konditioniert** die Funktion
ist und wie die **Schrittweite** gewählt wird – beides hier selbst gemessen statt nur behauptet.

**Einordnung in die Reihe:**

```
Gradientenabstieg (WURZEL)                       [DIESES STÜCK]
 └─ Newton-Verfahren                             [gebaut]
      └─ Quasi-Newton (BFGS/L-BFGS)               [gebaut]
           └─ Lagrange-Multiplikatoren/KKT        [gebaut]
                ├─ Straf-/Barriere-Verfahren      [gebaut]
                └─ SQP                            [gebaut]
                     └─ Innere-Punkte-Verfahren   [gebaut]
 └─ Stochastische Gradientenverfahren             [gebaut, letztes Stück]
```

**Ergebnis in Kürze:** Bei exakter Liniensuche auf einer Quadratik mit Konditionszahl $\kappa$
sagt die klassische Theorie eine Konvergenzrate von höchstens $\left(\frac{\kappa-1}{\kappa+1}\right)^2$
je Schritt voraus (Nocedal & Wright 2006). Die hier gemessene Rate bleibt bei **jeder** geprüften
Konditionszahl ($\kappa=2,10,50,200$) unter dieser Schranke – die Theorie hält, ist aber (wie die
Novikoff-Schranke im Perceptron-Stück) ein gültiges, aber grobes Worst-Case-Versprechen. Bei
fester Schrittweite gibt es eine **scharfe** Divergenzschwelle bei genau $\eta=2/\lambda_{\max}$:
knapp darunter ($\eta=0{,}099$) konvergiert der Abstieg (wenn auch langsam), knapp darüber
($\eta=0{,}101$) explodiert er um **26 Zehnerpotenzen** in nur 200 Schritten. Backtracking
(Armijo) braucht dagegen bis $\kappa=500$ weniger als die halbe Iterationszahl
der klassischen "sicheren" festen Schrittweite $1/\lambda_{\max}$ – und ist bei $\kappa=500$ und
$\kappa=2000$ der einzige der beiden Modi, der innerhalb des Budgets wirklich konvergiert (bei
$\kappa=2000$ ist der Vorsprung nur noch klein: 1.896 Schritte gegenüber dem Budget von 2.000).

## Warum dieses Problem

Mindestens vier bestehende Demos in diesem Portfolio rufen `scipy.optimize.minimize`
(SLSQP/trust-constr) als Black-Box-Löser auf – `truss-sizing-demo`, `truss-shape-demo`,
`slow-steaming-demo`, `nutrition-demo` –, ohne das Verfahren dahinter zu erklären. Diese Linie
liefert das fehlende Fundament: Was tut ein solcher Löser eigentlich, Schritt für Schritt?
Gradientenabstieg ist der einfachste Anfang – kein Newton, keine Nebenbedingungen, nur der
Gradient und eine Schrittweite.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| Exakte Liniensuche auf einer Quadratik reproduziert die geschlossene Schrittweiten-Formel | ✅ max. relative Abweichung $1{,}27\cdot10^{-16}$ (Maschinengenauigkeit) |
| Gemessene Konvergenzrate bleibt unter der theoretischen Schranke $\left(\frac{\kappa-1}{\kappa+1}\right)^2$ | ✅ bei allen vier geprüften Konditionszahlen |
| Feste Schrittweite oberhalb $2/\lambda_{\max}$ divergiert, unterhalb konvergiert sie | ✅ scharfe Schwelle exakt bestätigt |
| Backtracking (Armijo) braucht weniger Iterationen als die "sichere" feste Schrittweite $1/\lambda_{\max}$ | ✅ bei jeder geprüften Konditionszahl, mit wachsendem Vorsprung |
| Gradienten-Check gegen finite Differenzen unter $10^{-6}$ | ✅ Quadratik $1{,}93\cdot10^{-10}$, Rosenbrock $1{,}06\cdot10^{-10}$ |

## Befunde (gemessen, keine Behauptungen)

**Konvergenzrate: gemessen vs. theoretische Schranke** (exakte Liniensuche, $d=10$, 500 Schritte
Budget):

| Konditionszahl κ | Gemessene Rate/Schritt | Theoretische Schranke $\left(\frac{\kappa-1}{\kappa+1}\right)^2$ | Iterationen bis Konvergenz |
|---|---|---|---|
| 2 | 0,0830 | 0,1111 | 20 |
| 10 | 0,5700 | 0,6694 | 101 |
| 50 | 0,8287 | 0,9231 | 500 (Budget erreicht) |
| 200 | 0,8881 | 0,9802 | 500 (Budget erreicht) |

**Divergenzschwelle bei fester Schrittweite** ($d=6$, $\kappa=20$, Schwelle $2/\lambda_{\max}=0{,}1000$):

| Faktor × Schwelle | η | Divergiert? | f(x) nach 200 Schritten |
|---|---|---|---|
| 0,50 | 0,0500 | Nein | $2{,}12\cdot10^{-12}$ |
| 0,90 | 0,0900 | Nein | $7{,}13\cdot10^{-20}$ |
| 0,99 | 0,0990 | Nein | $2{,}56\cdot10^{-3}$ |
| 1,01 | 0,1010 | **Ja** | $2{,}28\cdot10^{4}$ |
| 1,10 | 0,1100 | **Ja** | $3{,}90\cdot10^{32}$ |
| 1,50 | 0,1500 | **Ja** | $2{,}14\cdot10^{121}$ |

**Backtracking vs. feste Schrittweite** ($d=10$, Budget 2000 Iterationen, feste Schrittweite
$=1/\lambda_{\max}$):

| κ | Iterationen (fest) | Iterationen (Backtracking) | f am Ende (fest) | f am Ende (Backtracking) |
|---|---|---|---|---|
| 5 | 89 | 43 | $4{,}73\cdot10^{-21}$ | $1{,}87\cdot10^{-21}$ |
| 20 | 386 | 131 | $4{,}88\cdot10^{-21}$ | $3{,}15\cdot10^{-21}$ |
| 100 | 1.969 | 887 | $4{,}98\cdot10^{-21}$ | $3{,}22\cdot10^{-21}$ |
| 500 | 2.000 (Budget) | 500 | $3{,}61\cdot10^{-7}$ | $3{,}89\cdot10^{-21}$ |
| 2.000 | 2.000 (Budget) | 1.896 | $1{,}03\cdot10^{-2}$ | $4{,}62\cdot10^{-21}$ |

Ab $\kappa=500$ erreicht die feste Schrittweite ihr Budget, **ohne wirklich konvergiert zu sein**
– Backtracking dagegen liegt bei jeder Konditionszahl auf Maschinengenauigkeit.

**Rosenbrock-Funktion** (Start $(-1{,}2,\ 1{,}0)$, Optimum $(1,1)$):

| Modus | Iterationen | Konvergiert? | f am Ende |
|---|---|---|---|
| Feste Schrittweite η=0,001 (Budget 5.000) | 5.000 (Budget) | Nein | $3{,}76\cdot10^{-3}$ |
| Backtracking (Armijo) | 767 | Ja | $1{,}19\cdot10^{-20}$ |

## Modell und Verfahren

- `gd_functions.py` – Testfunktionen: Quadratik $f(x)=\tfrac12 x^\top Ax$ mit kontrollierter
  Konditionszahl (Eigenwerte log-verteilt, zufällig rotiert) und Rosenbrock-Funktion; je
  Funktionswert, Gradient und Hesse-Matrix von Hand hergeleitet.
- `gd_optimizer.py` – `gradient_descent()` mit drei Schrittweiten-Modi: fest, exakte Liniensuche
  (geschlossene Formel), Backtracking/Armijo.
- `gd_evaluation.py` – Konditionszahl-Sweep, Divergenzschwelle, Backtracking-vs-fest-Vergleich,
  Korrektheits-Kette (exakte Liniensuche-Formel), Gradienten-Check.
- `gd_visualization.py` – Plotly: 2D-Kontur+Trajektorie, Konvergenzkurve, Konditionszahl-vs-Rate,
  Divergenz-Balken, Backtracking-vs-fest-Vergleich.

## Was die App zeigt

Funktion (Quadratik/Rosenbrock), Schrittweiten-Modus, Konditionszahl, feste Schrittweite, max.
Iterationen und Seed in der Sidebar; ein 2D-Kontur-Plot mit dem tatsächlichen Abstiegspfad und die
Konvergenzkurve für die aktuelle Konfiguration; der Konditionszahl-vs-Konvergenzrate-Sweep als
zentraler Befund; ein "📐"-Abschnitt mit der Korrektheits-Kette (exakte Liniensuche-Formel,
Divergenzschwelle, Backtracking-Vergleich, Gradienten-Check).

## Was nicht funktioniert hat / Grenzen

Bei dieser Vormessung bestätigte sich jede Vorab-Hypothese – keine Plan-Korrektur nötig. **Echte
Grenzen:** Nur unrestringierte Minimierung (Nebenbedingungen kommen erst ab Stück 4). Die
interaktive App hält die Quadratik bei Dimension 2 (damit die Kontur sichtbar bleibt); die
Konditionszahl- und Backtracking-Sweeps laufen intern bei $d=10$, um zu zeigen, dass der Effekt
nicht nur ein 2D-Artefakt ist. Kein Vergleich zu Newton/Quasi-Newton (kommt in den nächsten
Stücken) – dieses Stück zeigt Gradientenabstieg für sich, nicht im Wettbewerb.

## Tests

40 Tests, `python -m pytest tests/ -v` (Laufzeit lokal ~8 Sekunden – alle Sweeps sind reine,
schnelle lineare Algebra ohne Trainings-Chaotik):
- `test_functions.py` – Testfunktionen, Konditionszahl-Konstruktion, Gradienten gegen finite
  Differenzen.
- `test_optimizer.py` – alle drei Schrittweiten-Modi, Divergenz/Konvergenz.
- `test_evaluation.py` – Sweep-Funktionen mit billigen Parametern.
- `test_claims.py` – jede Zahl oben nachgerechnet, mit Toleranzband (Modul-Fixtures für die
  teureren Sweeps).
- `test_presets.py`, `test_app.py` – Presets, Regler-Extremwerte, Funktionswechsel, Footer.
- `test_oracle_gradient_descent.py` – unabhängige Orakel: Eigenzerlegung (Pfad und Schrittzahl bei fester
  Schrittweite), `scipy.optimize` (Liniensuche, Rosenbrock-Gradient/Hesse-Matrix, BFGS-Minimum),
  Kantorowitsch-Schranke je Schritt.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `gd_constants.py` | Regler-Grenzen, Presets |
| `gd_functions.py` | Testfunktionen (Quadratik, Rosenbrock) |
| `gd_optimizer.py` | Gradientenabstieg, drei Schrittweiten-Modi |
| `gd_evaluation.py` | Sweeps, Korrektheits-Kette, Gradienten-Check |
| `gd_visualization.py` | Plotly-Plots |
| `gd_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Kein Vergleich gegen SciPy/andere Solver – die Konvergenz wird gegen das **bekannte exakte
Optimum** geprüft ($x^*=0$ bzw. $(1,1)$), eine stärkere Garantie als ein Solver-Kreuzvergleich.
Keine höherdimensionale interaktive Kontur (bleibt bei $d=2$ für die Visualisierung) – die
höherdimensionalen Sweeps zeigen den allgemeinen Fall separat.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Cauchy, A.-L. (1847). *Méthode générale pour la résolution des systèmes d'équations
  simultanées.* Comptes Rendus de l'Académie des Sciences, 25, 536–538.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2. Aufl.). Springer.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Nichtlineare Optimierung: acht Stücke, zwei Äste](https://sebastianhanisch.net/konzepte-nichtlineare-optimierung.html).
