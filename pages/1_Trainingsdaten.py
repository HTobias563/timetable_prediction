import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import DURATION_COLS, PHASE_LABELS, TEIL_COLS

st.set_page_config(page_title="Trainingsdaten", layout="wide")

st.title("Trainingsdaten")
st.caption("Synthetischer Datensatz — 80 Projekte · 38 Spalten · Basis für die Modellkette")
st.divider()

@st.cache_data
def load_data():
    return pd.read_csv("pep_terminplan_synthetic.csv", sep=";")

df = load_data()

# ── Kennzahlen ────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Projekte", len(df))
c2.metric("Features (Input)", 19)
c3.metric("Zielvariablen", 7)
c4.metric("Auxiliary Target", 1)

st.divider()

# ── Filter + Tabelle ──────────────────────────────────────────────────────────

st.subheader("Datensatz")

fc1, fc2, fc3 = st.columns(3)
f_typ  = fc1.multiselect("Projekttyp",  df["projekttyp"].unique(),  df["projekttyp"].unique())
f_ant  = fc2.multiselect("Antriebsart", df["antriebsart"].unique(), df["antriebsart"].unique())
f_seg  = fc3.multiselect("Segment",     df["fahrzeugsegment"].unique(), df["fahrzeugsegment"].unique())

df_f = df[
    df["projekttyp"].isin(f_typ) &
    df["antriebsart"].isin(f_ant) &
    df["fahrzeugsegment"].isin(f_seg)
]

st.caption(f"{len(df_f)} von {len(df)} Projekten angezeigt")
st.dataframe(df_f, use_container_width=True, height=280)

st.divider()

# ── Verteilungen ──────────────────────────────────────────────────────────────

st.subheader("Verteilungen — Input Features")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df_f, x="projekttyp", color="projekttyp",
        title="Projekttyp", labels={"projekttyp": ""},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        df_f, x="antriebsart", color="antriebsart",
        title="Antriebsart", labels={"antriebsart": ""},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig = px.histogram(
        df_f, x="aehnlichkeit_vorgaenger_pct",
        title="Ähnlichkeit zum Vorgänger (%)",
        nbins=20, color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.histogram(
        df_f, x="ressourcen_fte",
        title="Ressourcen (FTE)", nbins=20,
        color_discrete_sequence=["#ff7f0e"],
    )
    fig.update_layout(height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Schwerpunktumfänge ────────────────────────────────────────────────────────

st.subheader("Schwerpunktumfänge — Anteil Neuentwicklungen je Bauteil")

teil_labels = [c.replace("teil_", "").replace("_", " ").title() for c in TEIL_COLS]
teil_means  = df_f[TEIL_COLS].mean().values * 100

fig = go.Figure(go.Bar(
    x=teil_means, y=teil_labels,
    orientation="h",
    marker_color="#2ca02c",
    text=[f"{v:.0f}%" for v in teil_means],
    textposition="outside",
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

col_l, col_r = st.columns(2)

with col_l:
    fig = go.Figure()
    for col, label, color in zip(DURATION_COLS, PHASE_LABELS,
                                  ["#1f77b4","#ff7f0e","#2ca02c",
                                   "#d62728","#9467bd","#8c564b","#e377c2"]):
        fig.add_trace(go.Box(
            y=df_f[col], name=label, marker_color=color, boxmean=True,
        ))
    fig.update_layout(
        title="Boxplot je Phase (Tage)",
        height=420, margin=dict(t=40, b=80),
        xaxis=dict(tickangle=-30),
        showlegend=False,
        plot_bgcolor="#f8f9fa",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    corr_cols = DURATION_COLS + ["aehnlichkeit_vorgaenger_pct",
                                  "ressourcen_fte", "anzahl_teile_neu"]
    corr = df_f[corr_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Korrelationsmatrix",
    )
    fig.update_layout(height=420, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
