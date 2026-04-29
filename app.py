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
    "Projektstart → Konzeptfreigabe",
    "KF → Designfreigabe",
    "DF → Techn. Freigabe",
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

TEIL_OPTIONS = {
    "Verbrennermotor":    "teil_verbrenner_motor",
    "E-Motor":            "teil_e_motor",
    "Getriebe":           "teil_getriebe",
    "Abgasanlage":        "teil_abgasanlage",
    "Batteriesystem":     "teil_batteriesystem",
    "Vorderachse":        "teil_vorderachse",
    "Hinterachse":        "teil_hinterachse",
    "Lenkung":            "teil_lenkung",
    "Rohbau":             "teil_rohbau",
    "Außendesign":        "teil_aussendesign",
    "Bordnetz":           "teil_bordnetz",
    "Steuergeräte (ECU)": "teil_steuergeraete_ecu",
    "Ladesystem":         "teil_ladesystem",
    "Cockpit / HMI":      "teil_cockpit_hmi",
    "Sitzsystem":         "teil_sitzsystem",
    "Kamera / Radar":     "teil_kamera_radar",
    "ADAS Software":      "teil_adas_software",
    "Kühlkreislauf":      "teil_kuehlkreislauf",
}

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
        handle_unknown="use_encoded_value", unknown_value=-1,
    )
    df_enc = df.copy()
    df_enc[CAT_FEATURES] = enc.fit_transform(df[CAT_FEATURES])
    X = df_enc[INPUT_FEATURES].copy()
    y_dur = df_enc[DURATION_COLS]
    y_aux = df_enc[AUXILIARY]
    models = {}
    feat_m1 = list(X.columns)
    m1 = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=6,
                              min_samples_leaf=3, random_state=42)
    )
    m1.fit(X, pd.concat([y_dur["dauer_start_kf_d"], y_aux], axis=1))
    models["dauer_start_kf_d"] = (m1, feat_m1, True)
    X["dauer_start_kf_d"] = y_dur["dauer_start_kf_d"].values
    X[AUXILIARY] = y_aux.values
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


def row_to_dict(row: pd.Series) -> dict:
    """CSV-Zeile in Feature-Dict umwandeln."""
    d = {}
    for key in [
        "projekttyp", "fahrzeugsegment", "antriebsart",
        "zielmarkt", "entwicklungsstandort",
        "geplante_stueckzahl", "anzahl_varianten",
        "aehnlichkeit_vorgaenger_pct", "ressourcen_fte",
        *TEIL_OPTIONS.values(),
    ]:
        d[key] = row.get(key, 0)
    teil_cols = list(TEIL_OPTIONS.values())
    d["anzahl_teile_neu"] = sum(int(d.get(c, 0)) for c in teil_cols)
    return d


def show_results(preds, projekt_start):
    durations  = [preds[c] for c in DURATION_COLS]
    cumulative = np.cumsum([0] + durations)
    total_days = int(cumulative[-1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Gesamtdauer bis SOP", f"{total_days} Tage",
              f"≈ {total_days/365:.1f} Jahre")
    c2.metric("SOP-Datum",
              (projekt_start + timedelta(days=total_days)).strftime("%d.%m.%Y"))
    c3.metric("Verfeinerte Stückzahl (nach KF)",
              f"{int(preds[AUXILIARY]):,}".replace(",", "."))

    st.divider()
    st.subheader("Terminplan")

    fig = go.Figure()
    for label, dur, start, color in zip(
        PHASE_LABELS, durations, cumulative[:-1], PHASE_COLORS
    ):
        sd = projekt_start + timedelta(days=int(start))
        ed = projekt_start + timedelta(days=int(start + dur))
        fig.add_trace(go.Bar(
            name=label, y=[label], x=[dur], base=[start],
            orientation="h", marker_color=color,
            text=f"{int(dur)} Tage", textposition="inside",
            insidetextanchor="middle",
            hovertemplate=(
                f"<b>{label}</b><br>Dauer: {int(dur)} Tage<br>"
                f"Start: {sd.strftime('%d.%m.%Y')}<br>"
                f"Ende: {ed.strftime('%d.%m.%Y')}<extra></extra>"
            ),
        ))

    for ms, cum in zip(MILESTONE_LABELS, cumulative):
        ms_date = projekt_start + timedelta(days=int(cum))
        fig.add_vline(x=cum, line_dash="dot", line_color="gray", line_width=1)
        fig.add_annotation(
            x=cum, y=1.02, yref="paper",
            text=f"<b>{ms}</b><br>{ms_date.strftime('%m/%Y')}",
            showarrow=False, font=dict(size=9), textangle=-45, xanchor="left",
        )

    fig.update_layout(
        barmode="overlay", showlegend=False, height=320,
        margin=dict(t=120, b=20, l=10, r=10),
        xaxis=dict(title="Tage ab Projektstart", range=[0, total_days * 1.05]),
        yaxis=dict(showticklabels=False),
        plot_bgcolor="#f8f9fa",
    )
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for label, dur, start in zip(PHASE_LABELS, durations, cumulative[:-1]):
        sd = projekt_start + timedelta(days=int(start))
        ed = projekt_start + timedelta(days=int(start + dur))
        rows.append({
            "Phase": label,
            "Dauer (Tage)": int(dur),
            "Start": sd.strftime("%d.%m.%Y"),
            "Meilenstein": ed.strftime("%d.%m.%Y"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── App ───────────────────────────────────────────────────────────────────────

with st.spinner("Modelle werden trainiert..."):
    models, enc = train_models()

st.title("PEP Timetable Prediction")
st.caption("Kaskadierende Modellkette · Synthetischer Proof-of-Concept")
st.divider()

tab_manual, tab_csv = st.tabs(["Manuelle Eingabe", "CSV Upload"])

# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — Manuelle Eingabe
# ════════════════════════════════════════════════════════════════════════════

with tab_manual:

    # Zeile 1: Kategoriale Features
    c1, c2, c3, c4, c5 = st.columns(5)
    projekttyp       = c1.selectbox("Projekttyp",          ["Neuanlauf", "Facelift", "Derivat"])
    fahrzeugsegment  = c2.selectbox("Fahrzeugsegment",     ["A", "B", "C", "D", "SUV", "Van"])
    antriebsart      = c3.selectbox("Antriebsart",         ["Verbrenner", "Hybrid", "PHEV", "Elektro"])
    zielmarkt        = c4.selectbox("Zielmarkt",           ["Europa", "USA", "China", "Global"])
    entwicklungsstandort = c5.selectbox("Entwicklungsstandort", ["Deutschland", "USA", "China", "Indien"])

    # Zeile 2: Numerische Features
    c1, c2, c3, c4, c5 = st.columns(5)
    geplante_stueckzahl = c1.number_input(
        "Stückzahl / Jahr", min_value=1_000, max_value=500_000, value=50_000, step=1_000
    )
    anzahl_varianten    = c2.number_input("Varianten",              min_value=1,  max_value=8,   value=2)
    aehnlichkeit        = c3.number_input("Ähnlichkeit Vorgänger (%)", min_value=0, max_value=100, value=40)
    ressourcen_fte      = c4.number_input("Ressourcen (FTE)",       min_value=20, max_value=300, value=100)
    projekt_start_m     = c5.date_input("Projektstart", value=date.today(), key="start_manual")

    # Zeile 3: Schwerpunktumfänge als Multiselect
    default_teile = []
    if projekttyp == "Neuanlauf":
        default_teile = ["Verbrennermotor", "Getriebe", "Rohbau", "Steuergeräte (ECU)", "Kühlkreislauf"]
        if antriebsart in ["Elektro", "PHEV", "Hybrid"]:
            default_teile += ["E-Motor", "Batteriesystem", "Ladesystem"]
    elif projekttyp == "Facelift":
        default_teile = ["Außendesign", "Cockpit / HMI"]

    selected_teile = st.multiselect(
        "Schwerpunktumfänge — betroffene Bauteile (Neuentwicklung)",
        options=list(TEIL_OPTIONS.keys()),
        default=default_teile,
    )

    teil_flags = {col: int(label in selected_teile)
                  for label, col in TEIL_OPTIONS.items()}

    if st.button("Terminplan berechnen", type="primary", key="btn_manual"):
        row = {
            "projekttyp": projekttyp, "fahrzeugsegment": fahrzeugsegment,
            "antriebsart": antriebsart, "zielmarkt": zielmarkt,
            "entwicklungsstandort": entwicklungsstandort,
            "geplante_stueckzahl": geplante_stueckzahl,
            "anzahl_varianten": anzahl_varianten,
            "aehnlichkeit_vorgaenger_pct": aehnlichkeit,
            "ressourcen_fte": ressourcen_fte,
            "anzahl_teile_neu": sum(teil_flags.values()),
            **teil_flags,
        }
        preds = predict(row, models, enc)
        st.divider()
        show_results(preds, projekt_start_m)

# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — CSV Upload
# ════════════════════════════════════════════════════════════════════════════

with tab_csv:

    st.markdown(
        "Lade eine CSV-Datei mit Projektdaten hoch. "
        "Die Spalten müssen dem Schema des Datensatzes entsprechen."
    )

    uploaded = st.file_uploader(
        "CSV hier ablegen oder auswählen",
        type=["csv"],
        help="Trennzeichen: Semikolon (;) oder Komma (,)",
    )

    if uploaded:
        try:
            raw = uploaded.read().decode("utf-8-sig")
            sep = ";" if raw.count(";") > raw.count(",") else ","
            df_upload = pd.read_csv(
                pd.io.common.StringIO(raw), sep=sep
            )
            st.success(f"{len(df_upload)} Projekt(e) geladen · {len(df_upload.columns)} Spalten")
            st.dataframe(df_upload.head(5), use_container_width=True)

            # Projekt auswählen wenn mehrere Zeilen
            if len(df_upload) > 1:
                id_col = "projekt_id" if "projekt_id" in df_upload.columns else None
                if id_col:
                    options = df_upload[id_col].tolist()
                    selected_id = st.selectbox("Projekt auswählen", options)
                    selected_row = df_upload[df_upload[id_col] == selected_id].iloc[0]
                else:
                    idx = st.number_input("Zeile auswählen (0-basiert)",
                                          min_value=0, max_value=len(df_upload)-1, value=0)
                    selected_row = df_upload.iloc[idx]
            else:
                selected_row = df_upload.iloc[0]

            projekt_start_csv = st.date_input("Projektstart", value=date.today(), key="start_csv")

            if st.button("Terminplan berechnen", type="primary", key="btn_csv"):
                row = row_to_dict(selected_row)
                preds = predict(row, models, enc)
                st.divider()
                show_results(preds, projekt_start_csv)

        except Exception as e:
            st.error(f"Fehler beim Lesen der CSV: {e}")
    else:
        st.info(
            "Erwartete Spalten: `projekttyp`, `fahrzeugsegment`, `antriebsart`, "
            "`zielmarkt`, `entwicklungsstandort`, `geplante_stueckzahl`, "
            "`anzahl_varianten`, `aehnlichkeit_vorgaenger_pct`, `ressourcen_fte`, "
            "sowie alle `teil_*` Flags (0/1)."
        )
        st.download_button(
            "Beispiel-CSV herunterladen",
            data=pd.read_csv("pep_terminplan_synthetic.csv", sep=";").head(3).to_csv(
                sep=";", index=False
            ).encode("utf-8-sig"),
            file_name="pep_beispiel.csv",
            mime="text/csv",
        )
