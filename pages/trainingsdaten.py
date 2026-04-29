import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import DURATION_COLS, PHASE_LABELS, TEIL_COLS

st.title("Trainingsdaten")
st.markdown(
    "Dieser Datensatz wurde **synthetisch generiert** und dient als Grundlage für das Training "
    "der kaskadierende Modellkette. Er enthält **80 fiktive Automotive-Projekte** mit realistischen "
    "Abhängigkeiten — z.B. dauern Neuanläufe länger als Facelifts, und Elektrofahrzeuge länger als Verbrenner."
)
st.divider()

@st.cache_data
def load_data():
    return pd.read_csv("pep_terminplan_synthetic.csv", sep=";")

df = load_data()

# ── Kennzahlen ────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Projekte (Zeilen)", len(df))
c2.metric("Input-Features", 19,  help="Am Projektstart bekannte Parameter")
c3.metric("Zielvariablen", 7,    help="Dauer je Meilenstein-Abschnitt in Tagen")
c4.metric("Auxiliary Target", 1, help="Verfeinerte Stückzahl nach Konzeptfreigabe")

st.divider()

# ── Vollständiger Datensatz ───────────────────────────────────────────────────

st.subheader("Vollständiger Datensatz")
st.markdown(
    "Jede Zeile ist ein Projekt. Die ersten Spalten sind die **Input-Features** (am T0 bekannt), "
    "die letzten Spalten sind die **Zielvariablen** — `dauer_*_d` = Tage zwischen zwei Meilensteinen."
)
st.dataframe(df, use_container_width=True, height=350)
st.caption(f"{len(df)} Zeilen · {len(df.columns)} Spalten · Trennzeichen: Semikolon")

st.divider()

# ── Verteilungen ──────────────────────────────────────────────────────────────

st.subheader("Verteilungen — Input-Features")
st.markdown(
    "Die kategorialen Features sind bewusst realistisch verteilt — "
    "mehr SUVs als Vans, mehr Facelifts als Neuanläufe."
)

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="projekttyp", color="projekttyp", title="Projekttyp",
                       color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(df, x="antriebsart", color="antriebsart", title="Antriebsart",
                       color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig = px.histogram(df, x="aehnlichkeit_vorgaenger_pct", nbins=20,
                       title="Ähnlichkeit zum Vorgänger (%)",
                       color_discrete_sequence=["#1f77b4"])
    fig.update_layout(height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.histogram(df, x="anzahl_teile_neu", nbins=15,
                       title="Anzahl neuer Bauteile je Projekt",
                       color_discrete_sequence=["#ff7f0e"])
    fig.update_layout(height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Schwerpunktumfänge ────────────────────────────────────────────────────────

st.subheader("Schwerpunktumfänge — Anteil Neuentwicklungen je Bauteil")
st.markdown(
    "Jedes `teil_*`-Flag ist **1**, wenn das Bauteil im jeweiligen Projekt neu entwickelt wird. "
    "Die Wahrscheinlichkeit ist nach Projekttyp abgestuft: "
    "Neuanläufe haben deutlich mehr neue Bauteile als Facelifts oder Derivate."
)

teil_labels = [c.replace("teil_", "").replace("_", " ").title() for c in TEIL_COLS]
teil_means  = df[TEIL_COLS].mean().values * 100

fig = go.Figure(go.Bar(
    x=teil_means, y=teil_labels, orientation="h",
    marker_color="#2ca02c",
    text=[f"{v:.0f}%" for v in teil_means], textposition="outside",
))
fig.update_layout(
    height=480, margin=dict(t=20, b=20, l=10, r=60),
    xaxis=dict(title="Anteil Projekte mit Neuentwicklung (%)", range=[0, 110]),
    yaxis=dict(autorange="reversed"),
    plot_bgcolor="#f8f9fa",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Zielvariablen ─────────────────────────────────────────────────────────────

st.subheader("Zielvariablen — Dauer je Meilenstein-Abschnitt")
st.markdown(
    "Die Zielvariablen sind die **Dauern in Tagen** zwischen je zwei Meilensteinen — "
    "nicht die absoluten Termine. Das macht die Werte projektübergreifend vergleichbar "
    "und ist die direkte Zielgröße der kaskadierende Modellkette."
)

col_l, col_r = st.columns(2)
with col_l:
    fig = go.Figure()
    for col, label, color in zip(DURATION_COLS, PHASE_LABELS,
                                  ["#1f77b4","#ff7f0e","#2ca02c",
                                   "#d62728","#9467bd","#8c564b","#e377c2"]):
        fig.add_trace(go.Box(y=df[col], name=label, marker_color=color, boxmean=True))
    fig.update_layout(
        title="Verteilung der Phasendauern (Tage)", height=400,
        margin=dict(t=40, b=80), xaxis=dict(tickangle=-30),
        showlegend=False, plot_bgcolor="#f8f9fa",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    corr_cols = DURATION_COLS + ["aehnlichkeit_vorgaenger_pct", "ressourcen_fte", "anzahl_teile_neu"]
    corr = df[corr_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Korrelationsmatrix")
    fig.update_layout(height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Die Korrelationsmatrix zeigt z.B. dass aufeinanderfolgende Phasen positiv korreliert sind — "
    "ein langer Projektstart geht oft mit einem insgesamt längeren Projekt einher."
)
