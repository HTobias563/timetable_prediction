import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import evaluate_models, DURATION_COLS, PHASE_LABELS, PHASE_COLORS

st.title("Modell-Evaluation")
st.markdown(
    "Hier siehst du, wie gut die kaskadierende Modellkette auf **ungesehenen Testdaten** abschneidet. "
    "Der Datensatz wurde 80/20 in Trainings- und Testdaten aufgeteilt — die Modelle wurden nur auf den "
    "Trainingsdaten trainiert und auf den Testdaten bewertet."
)
st.divider()

with st.spinner("Evaluation wird berechnet..."):
    results_df, feat_imp, sop_mae, sop_rmse, y_pred, y_true, n_train, n_test = evaluate_models()

# ── Feature Importance ────────────────────────────────────────────────────────

st.subheader("Feature Importance")
st.markdown(
    "Feature Importance zeigt, **welche Input-Features das Modell am stärksten nutzt**, "
    "um seine Vorhersage zu treffen. Ein hoher Wert bedeutet: dieses Feature hat großen Einfluss "
    "auf die vorhergesagte Phasendauer."
)

col_leg1, col_leg2 = st.columns(2)
col_leg1.info(
    "**Blau — Initialer Input-Feature**  \n"
    "Am Projektstart (T0) bekannte Parameter: z.B. `projekttyp`, `anzahl_teile_neu`, `ressourcen_fte`. "
    "Diese Werte liegen vor, bevor das erste Modell läuft."
)
col_leg2.error(
    "**Rot — Output eines Vorgänger-Modells**  \n"
    "Vorhersagen früherer Modelle in der Kette: z.B. `dauer_start_kf_d`, `stueckzahl_kf_refined`. "
    "Diese Werte sind am T0 **nicht** bekannt — sie werden erst durch die Kette erzeugt und dann "
    "als zusätzliche Features an das nächste Modell weitergegeben. "
    "Wenn ein roter Wert oben steht, heißt das: wie lange die Vorphase dauerte, ist der stärkste "
    "Prädiktor für die aktuelle Phase."
)

model_choice = st.selectbox(
    "Modell auswählen",
    options=[l for l, t in zip(PHASE_LABELS[1:], DURATION_COLS[1:]) if t in feat_imp],
    index=len(feat_imp) - 1,
)
target_key = DURATION_COLS[PHASE_LABELS.index(model_choice)]
imp = feat_imp[target_key].sort_values(ascending=True).tail(15)

colors = ["#d62728" if ("dauer_" in idx or idx == "stueckzahl_kf_refined")
          else "#1f77b4" for idx in imp.index]

fig = go.Figure(go.Bar(
    x=imp.values * 100, y=imp.index,
    orientation="h", marker_color=colors,
    text=[f"{v:.1f}%" for v in imp.values * 100], textposition="outside",
))
fig.add_annotation(
    x=imp.values.max() * 50, y=-1.8, yref="paper",
    text="🔴 = Output eines Vorgänger-Modells   🔵 = Initialer Input-Feature",
    showarrow=False, font=dict(size=11),
)
fig.update_layout(
    height=450, margin=dict(t=20, b=50, l=10, r=80),
    xaxis=dict(title="Importance (%)", range=[0, imp.values.max() * 130]),
    plot_bgcolor="#f8f9fa",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Übersicht Metriken ────────────────────────────────────────────────────────

st.subheader("Fehlermetriken — Überblick")
st.markdown(
    "**MAE (Mean Absolute Error)** = durchschnittlicher absoluter Fehler in Tagen. "
    "Leicht zu interpretieren: Im Schnitt weicht die Vorhersage um X Tage ab.  \n"
    "**RMSE (Root Mean Squared Error)** = bestraft große Ausreißer stärker als der MAE. "
    "Ein hoher RMSE bei niedrigem MAE bedeutet: vereinzelte Projekte werden sehr schlecht vorhergesagt."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trainings-Projekte", n_train)
c2.metric("Test-Projekte",      n_test)
c3.metric("Gesamt-SOP MAE",  f"{sop_mae} Tage",
          help="Summe aller 7 vorhergesagten Phasendauern vs. tatsächliche SOP-Dauer")
c4.metric("Gesamt-SOP RMSE", f"{sop_rmse} Tage")

st.divider()

# ── MAE-Tabelle + Balkendiagramm ──────────────────────────────────────────────

st.subheader("MAE & RMSE je Modell")
st.markdown(
    "Zwei MAE-Varianten werden verglichen:  \n"
    "- **Echte Vorwerte**: Jedes Modell bekommt die *wahren* Phasendauern der Vorgänger — "
    "zeigt die Güte des einzelnen Modells ohne Fehlerfortpflanzung.  \n"
    "- **Kask. Vorwerte**: Jedes Modell bekommt die *vorhergesagten* Vorgänger-Dauern — "
    "entspricht dem echten Einsatz, Fehler aus früheren Modellen fließen in spätere weiter."
)

col_t, col_b = st.columns([1, 1])
with col_t:
    st.dataframe(
        results_df.style.background_gradient(
            subset=["MAE (kask. Vorw.)", "RMSE (kask.)"], cmap="YlOrRd",
        ),
        hide_index=True, use_container_width=True, height=290,
    )
with col_b:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MAE (echte Vorwerte)", x=results_df["Phase"],
        y=results_df["MAE (echte Vorw.)"], marker_color="#1f77b4", opacity=0.6,
    ))
    fig.add_trace(go.Bar(
        name="MAE (kask. Vorwerte)", x=results_df["Phase"],
        y=results_df["MAE (kask. Vorw.)"], marker_color="#d62728",
    ))
    fig.update_layout(
        barmode="group", height=280, margin=dict(t=20, b=60, l=10, r=10),
        xaxis=dict(tickangle=-25), yaxis=dict(title="MAE (Tage)"),
        legend=dict(orientation="h", y=1.1), plot_bgcolor="#f8f9fa",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Predicted vs Actual ───────────────────────────────────────────────────────

st.subheader("Predicted vs. Actual je Phase")
st.markdown(
    "Jeder Punkt ist ein Test-Projekt. Punkte auf der **gestrichelten Diagonale** = perfekte Vorhersage. "
    "Punkte darüber = Modell überschätzt die Dauer, darunter = unterschätzt. "
    "Je enger die Punkte um die Diagonale, desto besser das Modell."
)

cols = st.columns(4)
for i, (target, label, color) in enumerate(zip(DURATION_COLS, PHASE_LABELS, PHASE_COLORS)):
    pred   = y_pred[target]
    true   = y_true[target]
    ax_max = max(np.max(pred), np.max(true)) * 1.1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=true, y=pred, mode="markers",
        marker=dict(color=color, size=8, opacity=0.8),
        hovertemplate="Ist: %{x:.0f} d<br>Pred: %{y:.0f} d<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, ax_max], y=[0, ax_max], mode="lines",
        line=dict(dash="dash", color="gray", width=1), showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=label, font=dict(size=11)), height=240,
        margin=dict(t=40, b=30, l=30, r=10),
        xaxis=dict(title="Ist (Tage)", range=[0, ax_max]),
        yaxis=dict(title="Pred (Tage)", range=[0, ax_max]),
        plot_bgcolor="#f8f9fa",
    )
    cols[i % 4].plotly_chart(fig, use_container_width=True)

st.divider()

# ── Fehlerfortpflanzung ───────────────────────────────────────────────────────

st.subheader("Fehlerfortpflanzung entlang der Kette")
st.markdown(
    "Da jedes Modell auf den Vorhersagen der Vorgänger aufbaut, **akkumuliert sich der Fehler** "
    "entlang der Kette. Die Kurve zeigt den kumulativen MAE nach jedem Schritt. "
    "Der Anstieg von Schritt zu Schritt zeigt, wie viel Fehler jede neue Phase hinzufügt — "
    "ein flacher Verlauf bedeutet, das Modell lernt gut trotz Fehlerfortpflanzung."
)

cum_true = np.zeros(n_test)
cum_pred = np.zeros(n_test)
cum_mae  = []
for target in DURATION_COLS:
    cum_true += y_true[target]
    cum_pred += y_pred[target]
    cum_mae.append(float(np.mean(np.abs(cum_true - cum_pred))))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=PHASE_LABELS, y=cum_mae, mode="lines+markers",
    line=dict(color="#d62728", width=2), marker=dict(size=9),
    fill="tozeroy", fillcolor="rgba(214,39,40,0.1)",
))
for label, val in zip(PHASE_LABELS, cum_mae):
    fig.add_annotation(x=label, y=val, text=f"{val:.0f} d",
                       showarrow=False, yshift=12, font=dict(size=10))
fig.update_layout(
    height=320, margin=dict(t=20, b=60, l=10, r=10),
    xaxis=dict(tickangle=-20), yaxis=dict(title="Kumulativer MAE (Tage)"),
    plot_bgcolor="#f8f9fa",
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Der finale kumulative MAE von {cum_mae[-1]:.0f} Tagen entspricht dem Gesamt-SOP-MAE: "
    "die Summe aller vorhergesagten Phasendauern weicht im Schnitt um diese Menge "
    "von der tatsächlichen Gesamtdauer ab."
)
