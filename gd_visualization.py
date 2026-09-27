"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe. Achsen fest uebergeben (siehe
feedback_plotly_fixedrange_convention/feedback_plotly_scaleanchor_explicit_range)."""
import numpy as np
import plotly.graph_objects as go

COLOR_PATH = "#1f77b4"
COLOR_START = "#d62728"
COLOR_OPT = "#2ca02c"
COLOR_MEASURED = "#1f77b4"
COLOR_THEORY = "#d62728"


def build_trajectory_figure(f, trajectory, x_star, x_range, y_range, title=""):
    xs = np.linspace(x_range[0], x_range[1], 120)
    ys = np.linspace(y_range[0], y_range[1], 120)
    Z = np.zeros((len(ys), len(xs)))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            Z[i, j] = f(np.array([xv, yv]))
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=np.log1p(Z), showscale=False, colorscale="Blues",
        contours=dict(coloring="fill"), opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        x=trajectory[:, 0], y=trajectory[:, 1], mode="lines+markers", name="Pfad",
        line=dict(color=COLOR_PATH, width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=[trajectory[0, 0]], y=[trajectory[0, 1]], mode="markers", name="Start",
        marker=dict(color=COLOR_START, size=12, symbol="x"),
    ))
    fig.add_trace(go.Scatter(
        x=[x_star[0]], y=[x_star[1]], mode="markers", name="Optimum",
        marker=dict(color=COLOR_OPT, size=13, symbol="star"),
    ))
    fig.update_layout(
        title=title, xaxis=dict(range=list(x_range), fixedrange=True, title="x₁"),
        yaxis=dict(range=list(y_range), fixedrange=True, title="x₂"),
        showlegend=True, height=420, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_convergence_figure(fvals, f_star=0.0, title="Konvergenz: f(xₖ) - f*"):
    gap = np.maximum(np.array(fvals) - f_star, 1e-300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(gap))), y=gap, mode="lines", name="f(xₖ) - f*",
        line=dict(color=COLOR_PATH, width=2),
    ))
    fig.update_layout(
        title=title, xaxis=dict(title="Iteration k", fixedrange=True),
        yaxis=dict(title="f(xₖ) - f*", fixedrange=True, type="log"),
        showlegend=False, height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_condition_sweep_figure(rows, title="Konvergenzrate: gemessen vs. theoretische Schranke"):
    kappas = [r["kappa"] for r in rows]
    measured = [r["measured_rate"] for r in rows]
    theory = [r["theoretical_rate"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kappas, y=theory, mode="lines+markers", name="Theoretische Schranke ((κ-1)/(κ+1))²",
        line=dict(color=COLOR_THEORY, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=kappas, y=measured, mode="lines+markers", name="Gemessene Rate (exakte Liniensuche)",
        line=dict(color=COLOR_MEASURED, width=2),
    ))
    fig.update_layout(
        title=title, xaxis=dict(title="Konditionszahl κ", fixedrange=True, type="log"),
        yaxis=dict(title="Konvergenzrate je Schritt", fixedrange=True),
        height=360, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_divergence_figure(threshold_data, title="Feste Schrittweite: Konvergenz vs. Divergenz"):
    threshold = threshold_data["theoretical_threshold"]
    rows = threshold_data["rows"]
    etas = [r["eta"] for r in rows]
    f_finals = [min(r["f_final"], 1e300) for r in rows]
    colors = ["#2ca02c" if not r["diverged"] else "#d62728" for r in rows]
    bar_width = (max(etas) - min(etas)) / (2.5 * len(etas))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=etas, y=f_finals, marker=dict(color=colors), width=bar_width,
                         name="f nach 200 Schritten"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#5B6B80",
                  annotation_text="Schwelle 2/λmax", annotation_position="top")
    fig.update_layout(
        title=title, xaxis=dict(title="Feste Schrittweite η", fixedrange=True),
        yaxis=dict(title="f(x) nach 200 Schritten", fixedrange=True, type="log"),
        showlegend=False, height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_backtracking_comparison_figure(rows, title="Iterationen bis Konvergenz: fest vs. Backtracking"):
    kappas = [r["kappa"] for r in rows]
    fixed_iters = [r["fixed_iters"] for r in rows]
    bt_iters = [r["backtrack_iters"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kappas, y=fixed_iters, mode="lines+markers",
                             name="Feste Schrittweite (1/λmax)", line=dict(color=COLOR_THEORY, width=2)))
    fig.add_trace(go.Scatter(x=kappas, y=bt_iters, mode="lines+markers",
                             name="Backtracking (Armijo)", line=dict(color=COLOR_MEASURED, width=2)))
    fig.update_layout(
        title=title, xaxis=dict(title="Konditionszahl κ", fixedrange=True, type="log"),
        yaxis=dict(title="Iterationen bis Konvergenz", fixedrange=True, type="log"),
        height=360, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
