import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import evaluate_models, evaluate_markov_baseline, DURATION_COLS, PHASE_LABELS, PHASE_COLORS

st.title("Modell-Evaluation")
st.markdown(
    "Wie gut funktioniert das Modell auf **Daten, die es noch nie gesehen hat**? "
    "Hier siehst du die Ergebnisse auf einem unabhängigen Testdatensatz (20 % der Daten)."
)
st.divider()

with st.spinner("Evaluation wird berechnet..."):
    results_df, feat_imp, sop_mae, sop_rmse, y_pred, y_true, n_train, n_test = evaluate_models()
    markov_df, sop_mae_markov, y_pred_markov = evaluate_markov_baseline()

# ── Kernergebnis ──────────────────────────────────────────────────────────────

avg_duration = float(pd.DataFrame(y_true).sum(axis=1).mean())
accuracy_pct = round((1 - sop_mae / avg_duration) * 100, 1)
improvement  = round(sop_mae_markov - sop_mae, 1)

st.subheader("Kernergebnis")
k1, k2, k3 = st.columns(3)
k1.metric(
    "Genauigkeit des KI-Modells",
    f"{accuracy_pct} %",
    help=f"Das Modell schätzt den SOP-Termin im Schnitt auf {accuracy_pct}% genau "
         f"(bezogen auf die mittlere Projektdauer von {int(avg_duration)} Tagen).",
)
k2.metric(
    "Durchschnittliche Abweichung",
    f"{sop_mae} Tage",
    help="Im Schnitt weicht die vorhergesagte Gesamtdauer um diese viele Tage ab.",
)
k3.metric(
    "Besser als einfache Schätzung",
    f"+ {improvement} Tage",
    help="Um so viele Tage genauer als ein einfacher historischer Mittelwert je Projekttyp.",
)

st.success(
    f"Das KI-Modell schätzt den SOP-Termin mit einer **durchschnittlichen Abweichung von "
    f"{sop_mae} Tagen** — das entspricht einer Genauigkeit von **{accuracy_pct} %**. "
    f"Gegenüber einer einfachen Durchschnittsschätzung ist es **{improvement} Tage genauer**."
)
st.caption(f"Testdatensatz: {n_test} Projekte · Trainingsdatensatz: {n_train} Projekte")

st.divider()

# ── KI-Modell vs. Einfache Schätzung ─────────────────────────────────────────

st.subheader("KI-Modell vs. einfache Schätzung")
st.markdown(
    "Als Vergleich dient eine **einfache Schätzung**: für jede Phase wird der historische "
    "Durchschnittswert je Projekttyp verwendet — kein Lernalgorithmus, nur Vergangenheitswerte. "
    "Je mehr das KI-Modell darunter liegt, desto mehr leistet es."
)

m1, m2, m3 = st.columns(3)
m1.metric("Abweichung — KI-Modell",          f"{sop_mae} Tage")
m2.metric("Abweichung — Einfache Schätzung", f"{sop_mae_markov} Tage")
m3.metric("Verbesserung durch KI",           f"{improvement} Tage",
          delta=f"{improvement} Tage besser", delta_color="normal")

fig = go.Figure()
fig.add_trace(go.Bar(
    name="KI-Modell",
    x=results_df["Phase"],
    y=results_df["MAE (kask. Vorw.)"],
    marker_color="#1f77b4",
))
fig.add_trace(go.Bar(
    name="Einfache Schätzung",
    x=markov_df["Phase"],
    y=markov_df["MAE (Markov)"],
    marker_color="#aec7e8",
    opacity=0.8,
))
fig.update_layout(
    barmode="group", height=320, margin=dict(t=20, b=60, l=10, r=10),
    xaxis=dict(tickangle=-25), yaxis=dict(title="Abweichung (Tage)"),
    legend=dict(orientation="h", y=1.08), plot_bgcolor="#f8f9fa",
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Blaue Balken unter den hellblauen = KI-Modell ist genauer als die einfache Schätzung. "
    "Die einfache Schätzung kennt nur den Projekttyp — alle anderen Parameter (Antriebsart, "
    "Bauteile, Ressourcen …) werden ignoriert."
)

st.divider()

# ── Abweichung je Phase ───────────────────────────────────────────────────────

st.subheader("Abweichung je Entwicklungsphase")
st.markdown(
    "Wie genau ist das Modell **in jeder einzelnen Phase**? "
    "Die Tabelle zeigt zwei Varianten:  \n"
    "- **Mit echten Vorgängerwerten** — zeigt, wie gut das Modell allein ist  \n"
    "- **Mit vorhergesagten Vorgängerwerten** — wie im echten Einsatz, Fehler aus früheren Phasen fließen weiter"
)

col_t, col_b = st.columns([1, 1])
with col_t:
    display_df = results_df.rename(columns={
        "MAE (echte Vorw.)": "Abw. ideal (Tage)",
        "MAE (kask. Vorw.)": "Abw. realistisch (Tage)",
        "RMSE (kask.)":      "Max.-Ausreißer (Tage)",
        "Ø Dauer (Tage)":    "Ø Dauer (Tage)",
    })
    st.dataframe(
        display_df.style.background_gradient(
            subset=["Abw. realistisch (Tage)", "Max.-Ausreißer (Tage)"], cmap="YlOrRd",
        ),
        hide_index=True, use_container_width=True, height=290,
    )
with col_b:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Abw. ideal", x=results_df["Phase"],
        y=results_df["MAE (echte Vorw.)"], marker_color="#1f77b4", opacity=0.6,
    ))
    fig.add_trace(go.Bar(
        name="Abw. realistisch", x=results_df["Phase"],
        y=results_df["MAE (kask. Vorw.)"], marker_color="#d62728",
    ))
    fig.update_layout(
        barmode="group", height=280, margin=dict(t=20, b=60, l=10, r=10),
        xaxis=dict(tickangle=-25), yaxis=dict(title="Abweichung (Tage)"),
        legend=dict(orientation="h", y=1.1), plot_bgcolor="#f8f9fa",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Vorhergesagt vs. Tatsächlich ──────────────────────────────────────────────

st.subheader("Vorhergesagt vs. Tatsächlich")
st.markdown(
    "Jeder Punkt ist ein Testprojekt. Punkte **auf der gestrichelten Linie** = perfekte Vorhersage. "
    "Je enger die Punkte um die Linie, desto genauer das Modell."
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
        hovertemplate="Tatsächlich: %{x:.0f} T<br>Vorhersage: %{y:.0f} T<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, ax_max], y=[0, ax_max], mode="lines",
        line=dict(dash="dash", color="gray", width=1), showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=label, font=dict(size=11)), height=240,
        margin=dict(t=40, b=30, l=30, r=10),
        xaxis=dict(title="Tatsächlich (Tage)", range=[0, ax_max]),
        yaxis=dict(title="Vorhersage (Tage)", range=[0, ax_max]),
        plot_bgcolor="#f8f9fa",
    )
    cols[i % 4].plotly_chart(fig, use_container_width=True)

st.divider()

# ── Fehlerfortpflanzung ───────────────────────────────────────────────────────

st.subheader("Wie wächst der Fehler über die Kette?")
st.markdown(
    "Das Modell arbeitet in 7 Schritten — jede Phase baut auf der vorherigen auf. "
    "Fehler aus frühen Phasen können sich in späteren verstärken. "
    "Die Kurve zeigt, wie viel Abweichung sich bis zu jedem Schritt angesammelt hat."
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
    fig.add_annotation(x=label, y=val, text=f"{val:.0f} T",
                       showarrow=False, yshift=12, font=dict(size=10))
fig.update_layout(
    height=320, margin=dict(t=20, b=60, l=10, r=10),
    xaxis=dict(tickangle=-20), yaxis=dict(title="Kumulierte Abweichung (Tage)"),
    plot_bgcolor="#f8f9fa",
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Die finale Abweichung von {cum_mae[-1]:.0f} Tagen entspricht der Gesamtabweichung beim SOP-Termin."
)

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────

st.subheader("Welche Faktoren beeinflussen die Vorhersage am stärksten?")
st.markdown(
    "Für jede Phase zeigt das Modell, welche Eingabe-Parameter es am stärksten genutzt hat. "
    "Ein hoher Wert bedeutet: dieser Parameter hatte großen Einfluss auf die Vorhersage."
)

col_leg1, col_leg2 = st.columns(2)
col_leg1.info("**Blau — Parameter, die von Anfang an bekannt sind** (z.B. Projekttyp, Ressourcen)")
col_leg2.error("**Rot — Vorhersagen früherer Phasen** (z.B. wie lange Phase 1 dauerte — erst durch das Modell bekannt)")

model_choice = st.selectbox(
    "Phase auswählen",
    options=[l for l, t in zip(PHASE_LABELS, DURATION_COLS) if t in feat_imp],
    index=0,
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
fig.update_layout(
    height=450, margin=dict(t=20, b=50, l=10, r=80),
    xaxis=dict(title="Einfluss (%)", range=[0, imp.values.max() * 130]),
    plot_bgcolor="#f8f9fa",
)
st.plotly_chart(fig, use_container_width=True)
