"""Gradientenabstieg — der erste Schritt bergab

Sebastian Hanisch - Operations Research und Machine Learning

Erstes Stück (Wurzel) der "Nichtlineare Optimierung"-Reihe der "Konzepte"-Reihe:
Gradientenabstieg -> Newton-Verfahren -> Quasi-Newton (BFGS/L-BFGS) -> Lagrange/KKT ->
{Straf-/Barriere-Verfahren, SQP -> Innere-Punkte-Verfahren} + Stochastische Gradientenverfahren.
Diese App zeigt den einfachsten Weg, eine glatte Funktion zu minimieren: immer entlang des
negativen Gradienten (Cauchy 1847) - inklusive der klassischen Konvergenzrate bei exakter
Liniensuche, hier selbst gemessen statt nur zitiert.

Lauffähig mit: streamlit run app.py
"""
import numpy as np
import streamlit as st

import gd_constants as C
import gd_evaluation as ev
import gd_functions as fn
import gd_presets as pr
import gd_visualization as viz

st.set_page_config(page_title="Gradientenabstieg", layout="wide")

FUNC_LABELS = {C.FUNC_QUADRATIC: "Quadratik (κ regelbar)", C.FUNC_ROSENBROCK: "Rosenbrock-Funktion"}
MODE_LABELS = {C.MODE_FIXED: "Feste Schrittweite", C.MODE_BACKTRACKING: "Backtracking (Armijo)",
               C.MODE_EXACT: "Exakte Liniensuche"}


@st.cache_data(show_spinner=False)
def _run(func, mode, kappa, eta, max_iter, seed):
    """Nur bildbare Rueckgabewerte (kein Funktionsobjekt) - st.cache_data picklet das Ergebnis."""
    import gd_optimizer as opt
    if func == C.FUNC_QUADRATIC:
        A = fn.random_spd_matrix(C.DIM, kappa, seed)
        f, grad, hess = fn.quadratic(A)
        rng = np.random.default_rng(seed + 1)
        x0 = rng.normal(size=C.DIM)
        result = opt.gradient_descent(f, grad, x0, mode=mode, eta=eta, A=A, max_iter=max_iter)
        x_star = np.zeros(C.DIM)
    else:
        f, grad, hess = fn.rosenbrock()
        result = opt.gradient_descent(f, grad, np.array([-1.2, 1.0]), mode=mode, eta=eta,
                                      max_iter=max_iter)
        x_star = np.array([1.0, 1.0])
    return {"trajectory": result.trajectory, "fvals": result.fvals, "converged": result.converged,
            "n_iter": result.n_iter, "x_star": x_star, "f_star": 0.0}


def _rebuild_function(func, kappa, seed):
    if func == C.FUNC_QUADRATIC:
        A = fn.random_spd_matrix(C.DIM, kappa, seed)
        f, grad, hess = fn.quadratic(A)
        return f
    f, grad, hess = fn.rosenbrock()
    return f


@st.cache_data(show_spinner=False)
def _condition_sweep():
    return ev.condition_number_sweep()


@st.cache_data(show_spinner=False)
def _divergence_threshold():
    return ev.step_size_divergence_threshold()


@st.cache_data(show_spinner=False)
def _backtracking_vs_fixed():
    return ev.backtracking_vs_fixed()


@st.cache_data(show_spinner=False)
def _exact_line_search_check():
    return ev.exact_line_search_check()


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


st.title("⛰️ Gradientenabstieg — der erste Schritt bergab")
st.markdown(
    "Um eine glatte Funktion zu minimieren, ohne mehr über sie zu wissen als ihren Gradienten: "
    "gehe immer entlang des **negativen Gradienten** (Cauchy 1847) — bergab, so lange bis sich "
    "nichts mehr bessert. Wie schnell das geht, hängt stark davon ab, wie **schlecht "
    "konditioniert** die Funktion ist und wie die **Schrittweite** gewählt wird. Dieses Stück "
    "misst beides selbst, statt es nur zu behaupten."
)
st.caption(
    "Wurzel der 'Nichtlineare Optimierung'-Reihe. Folgestücke (alle gebaut): "
    "Newton-Verfahren, Quasi-Newton (BFGS/L-BFGS), Lagrange/KKT, Straf-/Barriere-Verfahren, "
    "SQP, Innere-Punkte-Verfahren, Stochastische Gradientenverfahren."
)

with st.expander("So funktioniert Gradientenabstieg", expanded=True):
    st.markdown(
        "1. Starte bei einem Punkt $x_0$.\n"
        "2. Berechne den Gradienten $\\nabla f(x_k)$ — die Richtung des steilsten Anstiegs.\n"
        "3. Gehe einen Schritt in die Gegenrichtung: $x_{k+1}=x_k-\\eta_k\\nabla f(x_k)$.\n"
        "4. Wiederhole, bis der Gradient (fast) null ist.\n\n"
        "Die Schrittweite $\\eta_k$ entscheidet über alles: zu klein ist langsam, zu groß "
        "divergiert. **Exakte Liniensuche** wählt bei einer Quadratik die mathematisch optimale "
        "Schrittweite direkt aus einer Formel; **Backtracking (Armijo)** sucht sie durch "
        "Probieren, funktioniert aber auch bei nichtquadratischen Funktionen."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                   use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    func = st.radio("Funktion", C.FUNCTIONS, format_func=lambda f: FUNC_LABELS[f],
                    key="widget_func", index=C.FUNCTIONS.index(ss["func"]),
                    on_change=pr.store_from_widget, args=("func",))
    ss["func"] = func

    available_modes = C.MODES if func == C.FUNC_QUADRATIC else (C.MODE_FIXED, C.MODE_BACKTRACKING)
    mode_value = ss["mode"] if ss["mode"] in available_modes else C.MODE_BACKTRACKING
    mode = st.radio("Schrittweiten-Modus", available_modes, format_func=lambda m: MODE_LABELS[m],
                    key="widget_mode", index=available_modes.index(mode_value),
                    on_change=pr.store_from_widget, args=("mode",))
    ss["mode"] = mode

    if func == C.FUNC_QUADRATIC:
        kappa = st.slider("Konditionszahl κ", C.KAPPA_MIN, C.KAPPA_MAX, ss["kappa"], step=0.5,
                          key="widget_kappa", on_change=pr.store_from_widget, args=("kappa",))
    else:
        kappa = ss["kappa"]
    ss["kappa"] = kappa

    if mode == C.MODE_FIXED:
        eta_lo, eta_hi = (C.ETA_QUADRATIC_MIN, C.ETA_QUADRATIC_MAX) if func == C.FUNC_QUADRATIC \
            else (C.ETA_ROSENBROCK_MIN, C.ETA_ROSENBROCK_MAX)
        eta_value = min(max(ss["eta"], eta_lo), eta_hi)
        eta = st.slider("Feste Schrittweite η", eta_lo, eta_hi, eta_value,
                        step=(eta_hi - eta_lo) / 200, key="widget_eta",
                        on_change=pr.store_from_widget, args=("eta",), format="%.4f")
    else:
        eta = ss["eta"]
    ss["eta"] = eta

    max_iter = st.slider("Max. Iterationen", C.MAX_ITER_MIN, C.MAX_ITER_MAX, ss["max_iter"],
                         step=10, key="widget_max_iter", on_change=pr.store_from_widget,
                         args=("max_iter",))
    ss["max_iter"] = max_iter
    seed = st.number_input("Seed", value=ss["seed"], step=1, key="widget_seed",
                           on_change=pr.store_from_widget, args=("seed",))
    ss["seed"] = seed
    st.button("🎲 Zufälliger Seed", on_click=pr.randomize_seed)

pr.sync_query_params(dict(func=func, mode=mode, kappa=kappa, eta=eta, max_iter=max_iter, seed=seed))

out = _run(func, mode, kappa, eta, int(max_iter), int(seed))
f = _rebuild_function(func, kappa, int(seed))

st.markdown("---")
st.subheader("🎯 Der Weg zum Minimum")
col_left, col_right = st.columns([3, 2])
with col_left:
    if func == C.FUNC_QUADRATIC:
        pad = max(1.0, float(np.abs(out["trajectory"]).max()) * 1.2)
        x_range = y_range = (-pad, pad)
    else:
        x_range, y_range = (-2.0, 2.0), (-1.0, 3.0)
    fig_traj = viz.build_trajectory_figure(
        f, out["trajectory"], out["x_star"], x_range, y_range,
        title=f"Pfad ({FUNC_LABELS[func]}, {MODE_LABELS[mode]})",
    )
    st.plotly_chart(fig_traj, key=f"traj_{func}_{mode}_{kappa}_{eta}_{max_iter}_{seed}",
                    use_container_width=True)
with col_right:
    fig_conv = viz.build_convergence_figure(out["fvals"], out["f_star"])
    st.plotly_chart(fig_conv, key=f"conv_{func}_{mode}_{kappa}_{eta}_{max_iter}_{seed}",
                    use_container_width=True)

st.subheader("🎯 Was am Ende steht")
m1, m2, m3 = st.columns(3)
m1.metric("Konvergiert?", "Ja" if out["converged"] else "Nein (Max. Iterationen erreicht)")
m2.metric("Iterationen", f"{out['n_iter']}")
m3.metric("f(x) am Ende", f"{out['fvals'][-1]:.2e}")

st.markdown("---")
st.subheader("🎯 Die zentrale Messung: gemessene vs. theoretische Konvergenzrate")
sweep = _condition_sweep()
st.plotly_chart(viz.build_condition_sweep_figure(sweep), key="condition_sweep_chart",
                use_container_width=True)
st.caption(
    "Bei exakter Liniensuche auf einer Quadratik mit Konditionszahl κ sagt die klassische Theorie "
    "eine Konvergenzrate von höchstens ((κ-1)/(κ+1))² je Schritt voraus. Die hier gemessene Rate "
    "bleibt bei jeder geprüften Konditionszahl unter dieser Schranke — die Theorie hält, ist aber "
    "(wie die Novikoff-Schranke beim Perceptron) ein Worst-Case-Versprechen, kein exakter Wert."
)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Feste Schrittweite unter der Schwelle 2/λmax | Oberhalb der Schwelle divergiert "
    "Gradientenabstieg katastrophal (siehe 📐-Abschnitt) | Backtracking-Liniensuche |\n"
    "| Nur Gradient bekannt, keine Krümmung | Konvergenz linear, nicht quadratisch, und stark "
    "konditionszahlabhängig | Newton-Verfahren (nächstes Stück) |\n"
    "| Keine Nebenbedingungen | Reine unrestringierte Minimierung | Lagrange/KKT (Stück 4) |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Update:** $x_{k+1}=x_k-\eta_k\nabla f(x_k)$.

**Exakte Liniensuche auf einer Quadratik** $f(x)=\tfrac12 x^\top A x$: die optimale Schrittweite
hat eine geschlossene Formel, $\eta_k=\dfrac{\nabla f(x_k)^\top\nabla f(x_k)}{\nabla
f(x_k)^\top A\,\nabla f(x_k)}$ — hier gegen eine numerische 1D-Minimierung geprüft.

**Divergenzschwelle bei fester Schrittweite:** $\eta<2/\lambda_{\max}(A)$ ist notwendig für
Konvergenz.
"""
    )
    err = _exact_line_search_check()
    st.metric("Exakte Liniensuche: max. relative Abweichung von der Formel", f"{err:.2e}")

    thr = _divergence_threshold()
    st.plotly_chart(viz.build_divergence_figure(thr), key="divergence_chart",
                    use_container_width=True)
    st.caption(
        f"Theoretische Schwelle hier: η < {thr['theoretical_threshold']:.4f}. Unterhalb "
        "konvergiert Gradientenabstieg, oberhalb divergiert er messbar (Balken rechts der "
        "gestrichelten Linie explodieren um viele Zehnerpotenzen)."
    )

    bt = _backtracking_vs_fixed()
    st.plotly_chart(viz.build_backtracking_comparison_figure(bt), key="backtracking_chart",
                    use_container_width=True)
    st.caption(
        "Backtracking (Armijo) braucht bis κ=500 weniger als die halbe Iterationszahl der klassischen "
        "'sicheren' festen Schrittweite 1/λmax; bei κ=500 und κ=2000 erreicht die feste Schrittweite "
        "ihr Budget, ohne zu konvergieren."
    )

    grad_err = _gradient_check()
    g1, g2 = st.columns(2)
    g1.metric("Gradienten-Check Quadratik", f"{grad_err['quadratic_max_rel_err']:.2e}")
    g2.metric("Gradienten-Check Rosenbrock", f"{grad_err['rosenbrock_max_rel_err']:.2e}")

    st.markdown(
        "**Literatur:** Cauchy, A.-L. (1847). *Méthode générale pour la résolution des systèmes "
        "d'équations simultanées.* Comptes Rendus de l'Académie des Sciences, 25, 536–538. — "
        "Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2. Aufl.). Springer."
    )
    st.caption(
        "Implementiert in `gd_functions.py` (Testfunktionen), `gd_optimizer.py` "
        "(Gradientenabstieg), `gd_evaluation.py` (Sweeps, Korrektheits-Kette), "
        "`gd_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Nichtlineare Optimierung: acht Stücke, zwei Äste](https://sebastianhanisch.net/konzepte-nichtlineare-optimierung.html)."
)
