import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import evaluate_models, DURATION_COLS, PHASE_LABELS, PHASE_COLORS

st.set_page_config(page_title="Evaluation", layout="wide")

st.title("Modell-Evaluation")
st.caption("Kaskadierende Random-Forest-Kette · 80/20 Train/Test-Split · Teacher Forcing")
st.divider()

with st.spinner("Evaluation wird berechnet..."):
    results_df, feat_imp, sop_mae, sop_rmse, y_pred, y_true, n_train, n_test = evaluate_models()

# ── Übersicht ─────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trainings-Projekte", n_train)
c2.metric("Test-Projekte",      n_test)
c3.metric("Gesamt-SOP MAE",     f"{sop_mae} Tage")
c4.metric("Gesamt-SOP RMSE",    f"{sop_rmse} Tage")

st.divider()

# ── MAE-Tabelle + Balkendiagramm ──────────────────────────────────────────────

st.subheader("MAE & RMSE je Modell")

col_t, col_b = st.columns([1, 1])

with col_t:
    st.dataframe(
        results_df.style.background_gradient(
            subset=["MAE (kask. Vorw.)", "RMSE (kask.)"],
            cmap="YlOrRd",
        ),
        hide_index=True,
        use_container_width=True,
        height=290,
    )

with col_b:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MAE (echte Vorw.)",
        x=results_df["Phase"],
        y=results_df["MAE (echte Vorw.)"],
        marker_color="#1f77b4",
        opacity=0.6,
    ))
    fig.add_trace(go.Bar(
        name="MAE (kask. Vorw.)",
        x=results_df["Phase"],
        y=results_df["MAE (kask. Vorw.)"],
        marker_color="#d62728",
    ))
    fig.update_layout(
        barmode="group", height=280,
        margin=dict(t=20, b=60, l=10, r=10),
        xaxis=dict(tickangle=-25),
        yaxis=dict(title="MAE (Tage)"),
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="#f8f9fa",
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "**Echte Vorwerte:** Jedes Modell bekommt die wahren Werte der Vorgänger-Phasen — obere Grenze der Modellgüte.  "
    "**Kask. Vorwerte:** Jedes Modell bekommt die vorhergesagten Werte — realistischer Einsatz."
)

st.divider()

# ── Predicted vs Actual ───────────────────────────────────────────────────────

st.subheader("Predicted vs. Actual je Phase")

cols = st.columns(4)
for i, (target, label, color) in enumerate(zip(DURATION_COLS, PHASE_LABELS, PHASE_COLORS)):
    pred = y_pred[target]
    true = y_true[target]
    ax_max = max(np.max(pred), np.max(true)) * 1.1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=true, y=pred, mode="markers",
        marker=dict(color=color, size=8, opacity=0.8),
        hovertemplate="Ist: %{x:.0f} d<br>Pred: %{y:.0f} d<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, ax_max], y=[0, ax_max],
        mode="lines", line=dict(dash="dash", color="gray", width=1),
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=label, font=dict(size=11)),
        height=240,
        margin=dict(t=40, b=30, l=30, r=10),
        xaxis=dict(title="Ist (Tage)", range=[0, ax_max]),
        yaxis=dict(title="Pred (Tage)", range=[0, ax_max]),
        plot_bgcolor="#f8f9fa",
    )
    cols[i % 4].plotly_chart(fig, use_container_width=True)

st.divider()

# ── Fehlerfortpflanzung ───────────────────────────────────────────────────────

st.subheader("Fehlerfortpflanzung entlang der Kette")
st.caption("Wie akkumuliert sich der MAE wenn Vorhersagen in den nächsten Schritt fließen?")

cum_true = np.zeros(n_test)
cum_pred = np.zeros(n_test)
cum_mae  = []

for target in DURATION_COLS:
    cum_true += y_true[target]
    cum_pred += y_pred[target]
    cum_mae.append(float(np.mean(np.abs(cum_true - cum_pred))))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=PHASE_LABELS, y=cum_mae,
    mode="lines+markers",
    line=dict(color="#d62728", width=2),
    marker=dict(size=9),
    fill="tozeroy",
    fillcolor="rgba(214,39,40,0.1)",
))
for i, (label, val) in enumerate(zip(PHASE_LABELS, cum_mae)):
    fig.add_annotation(
        x=label, y=val, text=f"{val:.0f} d",
        showarrow=False, yshift=12, font=dict(size=10),
    )
fig.update_layout(
    height=320,
    margin=dict(t=20, b=60, l=10, r=10),
    xaxis=dict(tickangle=-20),
    yaxis=dict(title="Kumulativer MAE (Tage)"),
    plot_bgcolor="#f8f9fa",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────

st.subheader("Feature Importance")

model_choice = st.selectbox(
    "Modell auswählen",
    options=[l for l, t in zip(PHASE_LABELS[1:], DURATION_COLS[1:]) if t in feat_imp],
    index=len(feat_imp) - 1,
)
target_key = DURATION_COLS[PHASE_LABELS.index(model_choice)]
imp = feat_imp[target_key].sort_values(ascending=True).tail(15)

fig = go.Figure(go.Bar(
    x=imp.values * 100,
    y=imp.index,
    orientation="h",
    marker_color="#1f77b4",
    text=[f"{v:.1f}%" for v in imp.values * 100],
    textposition="outside",
))
fig.update_layout(
    height=420,
    margin=dict(t=20, b=20, l=10, r=60),
    xaxis=dict(title="Importance (%)", range=[0, imp.values.max() * 120]),
    plot_bgcolor="#f8f9fa",
)
st.plotly_chart(fig, use_container_width=True)
