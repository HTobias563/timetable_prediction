import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OrdinalEncoder

st.set_page_config(page_title="PEP Timetable Prediction", layout="wide")

# ── Konstanten ────────────────────────────────────────────────────────────────

DURATION_COLS = [
    "dauer_start_kf_d", "dauer_kf_df_d", "dauer_df_tf_d",
    "dauer_tf_p1_d", "dauer_p1_p2_d", "dauer_p2_vs_d", "dauer_vs_sop_d",
]

PHASE_LABELS = [
    "Projektstart → Konzeptfreigabe (KF)",
    "KF → Designfreigabe (DF)",
    "DF → Techn. Freigabe (TF)",
    "TF → Prototyp 1",
    "Prototyp 1 → Prototyp 2",
    "Prototyp 2 → Vorserie",
    "Vorserie → SOP",
]

MILESTONE_LABELS = [
    "Projektstart", "Konzeptfreigabe", "Designfreigabe",
    "Techn. Freigabe", "Prototyp 1", "Prototyp 2", "Vorserie", "SOP",
]

PHASE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b", "#e377c2",
]

INPUT_FEATURES = [
    "projekttyp", "fahrzeugsegment", "antriebsart", "zielmarkt",
    "entwicklungsstandort", "geplante_stueckzahl", "anzahl_varianten",
    "aehnlichkeit_vorgaenger_pct", "ressourcen_fte",
    "teil_verbrenner_motor", "teil_e_motor", "teil_getriebe",
    "teil_abgasanlage", "teil_batteriesystem",
    "teil_vorderachse", "teil_hinterachse", "teil_lenkung",
    "teil_rohbau", "teil_aussendesign",
    "teil_bordnetz", "teil_steuergeraete_ecu", "teil_ladesystem",
    "teil_cockpit_hmi", "teil_sitzsystem",
    "teil_kamera_radar", "teil_adas_software",
    "teil_kuehlkreislauf", "anzahl_teile_neu",
]

CAT_FEATURES = [
    "projekttyp", "fahrzeugsegment", "antriebsart",
    "zielmarkt", "entwicklungsstandort",
]

AUXILIARY = "stueckzahl_kf_refined"

# ── Modell trainieren (gecacht) ───────────────────────────────────────────────

@st.cache_resource
def train_models():
    df = pd.read_csv("pep_terminplan_synthetic.csv", sep=";")

    enc = OrdinalEncoder(
        categories=[
            ["Derivat", "Facelift", "Neuanlauf"],
            ["A", "B", "C", "D", "SUV", "Van"],
            ["Verbrenner", "Hybrid", "PHEV", "Elektro"],
            ["Europa", "USA", "China", "Global"],
            ["Deutschland", "USA", "China", "Indien"],
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    df_enc = df.copy()
    df_enc[CAT_FEATURES] = enc.fit_transform(df[CAT_FEATURES])

    X = df_enc[INPUT_FEATURES].copy()
    y_dur = df_enc[DURATION_COLS]
    y_aux = df_enc[AUXILIARY]

    models = {}

    # Model 1: Multi-Output
    feat_m1 = list(X.columns)
    m1 = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=6,
                              min_samples_leaf=3, random_state=42)
    )
    m1.fit(X, pd.concat([y_dur["dauer_start_kf_d"], y_aux], axis=1))
    models["dauer_start_kf_d"] = (m1, feat_m1, True)

    X["dauer_start_kf_d"] = y_dur["dauer_start_kf_d"].values
    X[AUXILIARY] = y_aux.values

    # Modelle 2–7
    for target in DURATION_COLS[1:]:
        feat_names = list(X.columns)
        rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                                   min_samples_leaf=3, random_state=42)
        rf.fit(X, y_dur[target])
        models[target] = (rf, feat_names, False)
        X[target] = y_dur[target].values

    return models, enc


def predict(row_dict, models, enc):
    df_in = pd.DataFrame([row_dict])
    df_in[CAT_FEATURES] = enc.transform(df_in[CAT_FEATURES])

    teil_cols = [c for c in INPUT_FEATURES if c.startswith("teil_")]
    df_in["anzahl_teile_neu"] = df_in[teil_cols].sum(axis=1)

    X = df_in[INPUT_FEATURES].copy()
    preds = {}

    m1, feat_m1, _ = models["dauer_start_kf_d"]
    out = m1.predict(X[feat_m1])[0]
    preds["dauer_start_kf_d"] = max(10, out[0])
    preds[AUXILIARY] = max(0, out[1])

    X["dauer_start_kf_d"] = preds["dauer_start_kf_d"]
    X[AUXILIARY] = preds[AUXILIARY]

    for target in DURATION_COLS[1:]:
        rf, feat_names, _ = models[target]
        preds[target] = max(10, rf.predict(X[feat_names])[0])
        X[target] = preds[target]

    return preds


# ── Modell laden ──────────────────────────────────────────────────────────────

with st.spinner("Modelle werden trainiert..."):
    models, enc = train_models()

# ── Layout ────────────────────────────────────────────────────────────────────

st.title("PEP Timetable Prediction")
st.caption("Kaskadierende Modellkette · Synthetischer Proof-of-Concept · Bachelorarbeit")
st.divider()

sidebar = st.sidebar
sidebar.header("Initiale Projektparameter")

# ── Sidebar: Eckdaten ─────────────────────────────────────────────────────────

sidebar.subheader("Projekteckdaten")

projekttyp = sidebar.selectbox(
    "Projekttyp", ["Neuanlauf", "Facelift", "Derivat"]
)
fahrzeugsegment = sidebar.selectbox(
    "Fahrzeugsegment", ["A", "B", "C", "D", "SUV", "Van"]
)
antriebsart = sidebar.selectbox(
    "Antriebsart", ["Verbrenner", "Hybrid", "PHEV", "Elektro"]
)
zielmarkt = sidebar.selectbox(
    "Zielmarkt", ["Europa", "USA", "China", "Global"]
)
entwicklungsstandort = sidebar.selectbox(
    "Entwicklungsstandort", ["Deutschland", "USA", "China", "Indien"]
)

sidebar.subheader("Planungsdaten")

geplante_stueckzahl = sidebar.number_input(
    "Geplante Stückzahl / Jahr", min_value=1_000, max_value=500_000,
    value=50_000, step=1_000,
)
anzahl_varianten = sidebar.slider("Anzahl Varianten", 1, 8, 2)
aehnlichkeit = sidebar.slider("Ähnlichkeit zum Vorgänger (%)", 0, 100, 40)
ressourcen_fte = sidebar.slider("Ressourcen (FTE)", 20, 300, 100)

projekt_start = sidebar.date_input("Projektstart", value=date.today())

# ── Sidebar: Schwerpunktumfänge ───────────────────────────────────────────────

sidebar.subheader("Schwerpunktumfänge")

with sidebar.expander("Antrieb", expanded=True):
    t_verbrenner = st.checkbox("Verbrennermotor", value=projekttyp == "Neuanlauf")
    t_e_motor    = st.checkbox("E-Motor",         value=antriebsart in ["Elektro", "PHEV", "Hybrid"])
    t_getriebe   = st.checkbox("Getriebe",        value=projekttyp == "Neuanlauf")
    t_abgas      = st.checkbox("Abgasanlage",     value=False)
    t_batterie   = st.checkbox("Batteriesystem",  value=antriebsart in ["Elektro", "PHEV"])

with sidebar.expander("Fahrwerk"):
    t_voa   = st.checkbox("Vorderachse",  value=projekttyp == "Neuanlauf")
    t_hia   = st.checkbox("Hinterachse",  value=False)
    t_lenk  = st.checkbox("Lenkung",      value=False)

with sidebar.expander("Karosserie & Design"):
    t_rohbau  = st.checkbox("Rohbau",        value=projekttyp == "Neuanlauf")
    t_design  = st.checkbox("Außendesign",   value=projekttyp in ["Neuanlauf", "Facelift"])

with sidebar.expander("Elektrik / E/E"):
    t_bordnetz = st.checkbox("Bordnetz",          value=False)
    t_ecu      = st.checkbox("Steuergeräte (ECU)", value=projekttyp == "Neuanlauf")
    t_laden    = st.checkbox("Ladesystem",         value=antriebsart in ["Elektro", "PHEV"])

with sidebar.expander("Innenraum & HMI"):
    t_cockpit = st.checkbox("Cockpit / HMI", value=projekttyp in ["Neuanlauf", "Facelift"])
    t_sitze   = st.checkbox("Sitzsystem",    value=False)

with sidebar.expander("ADAS"):
    t_kamera = st.checkbox("Kamera / Radar",  value=False)
    t_adas   = st.checkbox("ADAS Software",   value=False)

with sidebar.expander("Thermomanagement"):
    t_kuehl = st.checkbox("Kühlkreislauf", value=projekttyp == "Neuanlauf")

predict_btn = sidebar.button("Terminplan berechnen", type="primary", use_container_width=True)

# ── Vorhersage ────────────────────────────────────────────────────────────────

teil_flags = {
    "teil_verbrenner_motor": int(t_verbrenner),
    "teil_e_motor":          int(t_e_motor),
    "teil_getriebe":         int(t_getriebe),
    "teil_abgasanlage":      int(t_abgas),
    "teil_batteriesystem":   int(t_batterie),
    "teil_vorderachse":      int(t_voa),
    "teil_hinterachse":      int(t_hia),
    "teil_lenkung":          int(t_lenk),
    "teil_rohbau":           int(t_rohbau),
    "teil_aussendesign":     int(t_design),
    "teil_bordnetz":         int(t_bordnetz),
    "teil_steuergeraete_ecu":int(t_ecu),
    "teil_ladesystem":       int(t_laden),
    "teil_cockpit_hmi":      int(t_cockpit),
    "teil_sitzsystem":       int(t_sitze),
    "teil_kamera_radar":     int(t_kamera),
    "teil_adas_software":    int(t_adas),
    "teil_kuehlkreislauf":   int(t_kuehl),
}

row = {
    "projekttyp":                projekttyp,
    "fahrzeugsegment":           fahrzeugsegment,
    "antriebsart":               antriebsart,
    "zielmarkt":                 zielmarkt,
    "entwicklungsstandort":      entwicklungsstandort,
    "geplante_stueckzahl":       geplante_stueckzahl,
    "anzahl_varianten":          anzahl_varianten,
    "aehnlichkeit_vorgaenger_pct": aehnlichkeit,
    "ressourcen_fte":            ressourcen_fte,
    "anzahl_teile_neu":          sum(teil_flags.values()),
    **teil_flags,
}

preds = predict(row, models, enc)

# ── Ergebnisse ────────────────────────────────────────────────────────────────

durations = [preds[c] for c in DURATION_COLS]
cumulative = np.cumsum([0] + durations)
total_days = int(cumulative[-1])

# Metriken oben
col1, col2, col3 = st.columns(3)
col1.metric("Gesamtdauer bis SOP", f"{total_days} Tage",
            f"≈ {total_days/365:.1f} Jahre")
col2.metric("SOP-Datum",
            (projekt_start + timedelta(days=total_days)).strftime("%d.%m.%Y"))
col3.metric("Verfeinerte Stückzahl (nach KF)",
            f"{int(preds[AUXILIARY]):,}".replace(",", "."),
            f"Plan: {geplante_stueckzahl:,}".replace(",", "."))

st.divider()

# ── Gantt-Chart ───────────────────────────────────────────────────────────────

st.subheader("Terminplan")

fig = go.Figure()

for i, (label, dur, start, color) in enumerate(
    zip(PHASE_LABELS, durations, cumulative[:-1], PHASE_COLORS)
):
    start_date = projekt_start + timedelta(days=int(start))
    end_date   = projekt_start + timedelta(days=int(start + dur))

    fig.add_trace(go.Bar(
        name=label,
        y=[label],
        x=[dur],
        base=[start],
        orientation="h",
        marker_color=color,
        text=f"{int(dur)} Tage",
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate=(
            f"<b>{label}</b><br>"
            f"Dauer: {int(dur)} Tage<br>"
            f"Start: {start_date.strftime('%d.%m.%Y')}<br>"
            f"Ende: {end_date.strftime('%d.%m.%Y')}<extra></extra>"
        ),
    ))

# Milestone-Linien
for i, (ms, cum) in enumerate(zip(MILESTONE_LABELS, cumulative)):
    ms_date = projekt_start + timedelta(days=int(cum))
    fig.add_vline(
        x=cum, line_dash="dot", line_color="gray", line_width=1,
    )
    fig.add_annotation(
        x=cum, y=1.02, yref="paper",
        text=f"<b>{ms}</b><br>{ms_date.strftime('%m/%Y')}",
        showarrow=False, font=dict(size=9), textangle=-45,
        xanchor="left",
    )

fig.update_layout(
    barmode="overlay",
    showlegend=False,
    height=320,
    margin=dict(t=120, b=20, l=10, r=10),
    xaxis=dict(
        title="Tage ab Projektstart",
        range=[0, total_days * 1.05],
    ),
    yaxis=dict(showticklabels=False),
    plot_bgcolor="#f8f9fa",
)

st.plotly_chart(fig, use_container_width=True)

# ── Detailtabelle ─────────────────────────────────────────────────────────────

st.subheader("Details je Phase")

rows = []
for label, dur, start in zip(PHASE_LABELS, durations, cumulative[:-1]):
    start_date = projekt_start + timedelta(days=int(start))
    end_date   = projekt_start + timedelta(days=int(start + dur))
    rows.append({
        "Phase":         label,
        "Dauer (Tage)":  int(dur),
        "Start":         start_date.strftime("%d.%m.%Y"),
        "Ende / Meilenstein": end_date.strftime("%d.%m.%Y"),
    })

st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    use_container_width=True,
)

# ── Eingesetzte Umfänge ───────────────────────────────────────────────────────

active = [k.replace("teil_", "").replace("_", " ").title()
          for k, v in teil_flags.items() if v == 1]

if active:
    st.subheader("Aktive Schwerpunktumfänge")
    cols = st.columns(4)
    for i, name in enumerate(active):
        cols[i % 4].success(f"✓ {name}")
