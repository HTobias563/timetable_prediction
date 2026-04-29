import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model import PHASE_LABELS, PHASE_COLORS, DURATION_COLS

st.title("Methodik")
st.markdown("Wie funktionieren die beiden Modelle — und was unterscheidet sie?")
st.divider()

# ── 1. Kaskadierende RF-Kette ─────────────────────────────────────────────────

st.subheader("1 · Kaskadierende Random-Forest-Modellkette")

col_exp, col_fig = st.columns([1, 1])

with col_exp:
    st.markdown("""
**Was ist ein Random Forest?**

Ein Random Forest besteht aus vielen Entscheidungsbäumen.
Jeder Baum lernt an leicht unterschiedlichen Ausschnitten der Daten und
trifft seine eigene Vorhersage — das Ergebnis ist der Durchschnitt aller Bäume.
Dadurch werden Ausreißer abgefedert und das Modell wird stabiler.

---

**Warum eine Kaskade?**

Die 7 Phasen werden nicht gleichzeitig vorhergesagt,
sondern nacheinander — jede Phase baut auf den Vorhersagen der vorherigen auf:

- **Modell 1** kennt nur die 28 Projektparameter vom Start (T0) und sagt die erste Phase voraus.
- **Modell 2** bekommt dieselben 28 T0-Parameter *plus* die Vorhersage von Modell 1 — also 30 Features.
- Jedes weitere Modell wächst um eine Feature: es "weiß", wie lange die Vorphasen dauern.
- **Modell 7** arbeitet schließlich mit 35 Features.

**Die Idee dahinter:** In der Realität werden spätere Phasen von frühen Phasen beeinflusst —
wer lange in der Konzeptphase braucht, braucht oft auch länger danach.
Die Kaskade macht diesen Zusammenhang explizit nutzbar.
""")

with col_fig:
    step   = 1.5
    y_t0   = 7 * step + 1.5   # = 12.0
    feats  = [28, 30, 31, 32, 33, 34, 35]

    fig = go.Figure()

    # T0 box
    fig.add_shape(type="rect", x0=0, x1=5, y0=y_t0 - 0.4, y1=y_t0 + 0.4,
                  fillcolor="#dceefb", line=dict(color="#1f77b4", width=2))
    fig.add_annotation(x=2.5, y=y_t0,
                       text="<b>T0-Features</b>  (28 Parameter)",
                       showarrow=False, font=dict(size=10, color="#1f77b4"))

    for i in range(7):
        y     = y_t0 - step * (i + 1)
        color = PHASE_COLORS[i]

        fig.add_shape(type="rect", x0=0, x1=5, y0=y - 0.4, y1=y + 0.4,
                      fillcolor=color, opacity=0.18,
                      line=dict(color=color, width=2))
        fig.add_annotation(x=2.5, y=y + 0.12,
                           text=f"<b>Modell {i + 1}</b>",
                           showarrow=False, font=dict(size=10))
        fig.add_annotation(x=2.5, y=y - 0.16,
                           text=PHASE_LABELS[i],
                           showarrow=False, font=dict(size=9, color="#555"))
        fig.add_annotation(x=5.4, y=y,
                           text=f"{feats[i]} Features",
                           showarrow=False, font=dict(size=8, color="#888"),
                           xanchor="left")

        # Arrow from previous box
        from_y = y_t0 - 0.4 if i == 0 else y_t0 - step * i - 0.4
        fig.add_annotation(
            x=2.5, y=y + 0.4,
            ax=2.5, ay=from_y,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowwidth=1.5, arrowcolor="#888",
            showarrow=True, text="",
        )

    # SOP output box
    y_sop = y_t0 - step * 8
    fig.add_shape(type="rect", x0=0, x1=5, y0=y_sop - 0.4, y1=y_sop + 0.4,
                  fillcolor="#d5f5e3", line=dict(color="#2ca02c", width=2))
    fig.add_annotation(x=2.5, y=y_sop,
                       text="<b>SOP-Terminplan</b>",
                       showarrow=False, font=dict(size=10, color="#2ca02c"))
    fig.add_annotation(
        x=2.5, y=y_sop + 0.4,
        ax=2.5, ay=y_t0 - step * 7 - 0.4,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=2, arrowwidth=1.5, arrowcolor="#2ca02c",
        showarrow=True, text="",
    )

    fig.update_layout(
        height=600,
        xaxis=dict(visible=False, range=[-0.5, 9]),
        yaxis=dict(visible=False, range=[y_sop - 0.8, y_t0 + 0.8]),
        margin=dict(t=10, b=10, l=10, r=60),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# Teacher Forcing
with st.expander("Wie wird die Kette trainiert? — Teacher Forcing"):
    st.markdown("""
Beim Training bekommt jedes Modell die **echten** Vorgängerwerte aus dem Datensatz —
nicht die Vorhersagen der anderen Modelle. Das nennt sich **Teacher Forcing**:
der Lehrer (echte Daten) gibt die richtigen Antworten vor.

Bei der **Vorhersage** gibt es diese echten Werte nicht — jedes Modell bekommt
die Vorhersage des Vorgängers. Das ist der Grund, warum der Fehler sich in der
Kette aufbaut: ein Fehler in Modell 2 fließt direkt in Modell 3, 4, … weiter.

In der Evaluation werden deshalb zwei MAE-Varianten verglichen:
- **Echte Vorwerte** = Modell allein, ohne Fehlerfortpflanzung
- **Kask. Vorwerte** = realer Einsatz mit Fehlerfortpflanzung
""")

st.divider()

# ── 2. Markov-Baseline ────────────────────────────────────────────────────────

st.subheader("2 · Markov-Baseline (semi-Markov-Prozess)")

col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    st.markdown("""
**Was ist ein Markov-Prozess?**

Ein Markov-Prozess beschreibt Übergänge zwischen Zuständen.
Die Kernidee: **Der nächste Zustand hängt nur vom aktuellen ab — nicht von der Geschichte.**

Im PEP-Kontext:
- Die **Zustände** sind die Meilensteine (KF, DF, TF, …)
- Die **Übergänge** sind deterministisch — es gibt nur eine Reihenfolge
- Die **Verweildauer** in jedem Zustand (= Phasendauer) wird aus historischen Daten geschätzt

Das nennt sich **semi-Markov**: der Prozess ist Markov für die Übergänge,
aber die Verweildauer hat eine eigene Verteilung.

**Was lernt die Baseline hier?**

Für jede Kombination aus Projekttyp und Phase wird der
**historische Mittelwert** der Phasendauer berechnet.
Das ist die gesamte "Intelligenz" des Modells.

Das Diagramm rechts zeigt genau diese Werte — es ist das trainierte Markov-Modell.
""")
    st.info(
        "Kein einziger anderer Feature-Wert fließt ein — weder Antriebsart, "
        "Bauteil-Flags noch Ressourcen. Nur der Projekttyp zählt."
    )

with col_m2:
    df = pd.read_csv("pep_terminplan_synthetic.csv", sep=";")
    means = df.groupby("projekttyp")[DURATION_COLS].mean()

    fig = go.Figure()
    pt_colors = {"Neuanlauf": "#d62728", "Facelift": "#ff7f0e", "Derivat": "#1f77b4"}

    for pt, color in pt_colors.items():
        if pt in means.index:
            fig.add_trace(go.Bar(
                name=pt,
                x=PHASE_LABELS,
                y=means.loc[pt].values,
                marker_color=color,
                opacity=0.8,
            ))

    fig.update_layout(
        barmode="group", height=380,
        margin=dict(t=30, b=80, l=10, r=10),
        xaxis=dict(tickangle=-30),
        yaxis=dict(title="Mittlere Dauer (Tage)"),
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="#f8f9fa",
        title=dict(text="Markov-Modell: historische Mittelwerte je Projekttyp", font=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── 3. Vergleich ──────────────────────────────────────────────────────────────

st.subheader("3 · Vergleich der Ansätze")

vergleich = pd.DataFrame([
    ("Features",            "Alle 28 T0-Parameter",         "Nur Projekttyp"),
    ("Algorithmus",         "Random Forest (200 Bäume)",    "Historischer Mittelwert"),
    ("Lernfähigkeit",       "Lernt Zusammenhänge",          "Keine — reine Statistik"),
    ("Fehlerfortpflanzung", "Ja — Fehler akkumulieren sich","Nein — jede Phase unabhängig"),
    ("Interpretierbarkeit", "Feature Importance",           "Direkt ablesbar aus Daten"),
    ("Rolle im POC",        "Hauptmodell",                  "Benchmark / Untergrenze"),
], columns=["Kriterium", "RF-Kaskade", "Markov-Baseline"])

st.dataframe(vergleich, hide_index=True, use_container_width=True)

st.markdown("""
**Warum überhaupt eine Baseline?**

Die Baseline beantwortet die Frage: *Wie viel bringt das komplexere Modell wirklich?*
Wenn die RF-Kaskade kaum besser ist als der Mittelwert je Projekttyp,
lohnt sich die aufwändige Feature-Erhebung nicht.
Je größer der MAE-Unterschied, desto mehr rechtfertigt das Modell den Aufwand.
""")
