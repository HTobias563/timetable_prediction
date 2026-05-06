import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from model import (
    train_models, predict, row_to_dict,
    train_markov_baseline, predict_markov,
    DURATION_COLS, PHASE_LABELS, MILESTONE_LABELS,
    PHASE_COLORS, TEIL_OPTIONS, TEIL_COLS, AUXILIARY,
)

with st.spinner("Modelle werden geladen..."):
    models, enc          = train_models()
    markov_means, markov_overall = train_markov_baseline()

st.title("Terminplan-Vorhersage")
st.divider()

modell_wahl = st.radio(
    "Vorhersagemethode",
    options=["KI-Modell (Random Forest)", "Historischer Mittelwert je Projekttyp"],
    horizontal=True,
    help=(
        "KI-Modell: Nutzt alle eingegebenen Parameter für die Vorhersage.  \n"
        "Historischer Mittelwert: Schätzt anhand von Vergangenheitsdurchschnitten — "
        "nur der Projekttyp zählt, alle anderen Eingaben werden ignoriert."
    ),
)
use_rf = modell_wahl == "KI-Modell (Random Forest)"

if use_rf:
    st.markdown(
        "Die kaskadierende Modellkette berechnet aus den initialen Projektparametern "
        "den voraussichtlichen Terminplan bis zum **Start of Production (SOP)**."
    )
else:
    st.info(
        "**Historischer Mittelwert:** Vorhersage = durchschnittliche Dauer je Projekttyp aus den Trainingsdaten. "
        "Nur der `Projekttyp` wird ausgewertet — alle anderen Eingaben haben keinen Einfluss auf dieses Modell."
    )

st.divider()


def _predict(row_dict):
    if use_rf:
        return predict(row_dict, models, enc)
    return predict_markov(row_dict, markov_means, markov_overall)


def _fmt_duration(days: int) -> str:
    y, rest = divmod(days, 365)
    m = rest // 30
    if y and m:
        return f"{y} Jahre {m} Monate"
    return f"{y} Jahre" if y else f"{m} Monate"


def show_results(preds, projekt_start):
    durations  = [preds[c] for c in DURATION_COLS]
    cumulative = np.cumsum([0] + durations)
    total_days = int(cumulative[-1])
    sop_date   = projekt_start + timedelta(days=total_days)

    st.subheader("Ergebnis")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Gesamtdauer bis SOP",
        _fmt_duration(total_days),
        f"{total_days} Tage",
    )
    c2.metric("Voraussichtlicher SOP", sop_date.strftime("%d. %B %Y"))
    c3.metric(
        "Stückzahl nach Konzeptfreigabe",
        f"{int(preds[AUXILIARY]):,}".replace(",", "."),
    )

    st.divider()
    st.subheader("Terminplan")

    fig = go.Figure()
    for label, dur, start, color in zip(PHASE_LABELS, durations, cumulative[:-1], PHASE_COLORS):
        sd = projekt_start + timedelta(days=int(start))
        ed = projekt_start + timedelta(days=int(start + dur))
        fig.add_trace(go.Bar(
            name=label, y=[label], x=[dur], base=[start],
            orientation="h", marker_color=color,
            text=f"{int(dur)} Tage", textposition="inside", insidetextanchor="middle",
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
            "Phase":          label,
            "Dauer":          _fmt_duration(int(dur)),
            "Dauer (Tage)":   int(dur),
            "Start":          sd.strftime("%d.%m.%Y"),
            "Meilenstein":    ed.strftime("%d.%m.%Y"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


tab_manual, tab_csv = st.tabs(["Manuelle Eingabe", "CSV Upload"])

with tab_manual:
    c1, c2, c3, c4, c5 = st.columns(5)
    projekttyp           = c1.selectbox("Projekttyp",           ["Neuanlauf", "Facelift", "Derivat"])
    fahrzeugsegment      = c2.selectbox("Fahrzeugsegment",      ["A", "B", "C", "D", "SUV", "Van"])
    antriebsart          = c3.selectbox("Antriebsart",          ["Verbrenner", "Hybrid", "PHEV", "Elektro"])
    zielmarkt            = c4.selectbox("Zielmarkt",            ["Europa", "USA", "China", "Global"])
    entwicklungsstandort = c5.selectbox("Entwicklungsstandort", ["Deutschland", "USA", "China", "Indien"])

    c1, c2, c3, c4, c5 = st.columns(5)
    geplante_stueckzahl = c1.number_input("Stückzahl / Jahr",           1_000, 500_000, 50_000, 1_000)
    anzahl_varianten    = c2.number_input("Varianten",                  1, 8,   2)
    aehnlichkeit        = c3.number_input("Ähnlichkeit Vorgänger (%)", 0, 100, 40)
    ressourcen_fte      = c4.number_input("Ressourcen (FTE)",          20, 300, 100)
    projekt_start_m     = c5.date_input("Projektstart", value=date.today(), key="ps_m")

    default_teile = []
    if projekttyp == "Neuanlauf":
        default_teile = ["Verbrennermotor", "Getriebe", "Rohbau", "Steuergeräte (ECU)", "Kühlkreislauf"]
        if antriebsart in ["Elektro", "PHEV", "Hybrid"]:
            default_teile += ["E-Motor", "Batteriesystem", "Ladesystem"]
    elif projekttyp == "Facelift":
        default_teile = ["Außendesign", "Cockpit / HMI"]

    selected_teile = st.multiselect(
        "Neu entwickelte Bauteile",
        options=list(TEIL_OPTIONS.keys()), default=default_teile,
        help="Welche Bauteile werden in diesem Projekt neu entwickelt? "
             "Mehr Neuentwicklungen bedeuten in der Regel längere Phasen.",
    )
    teil_flags = {col: int(label in selected_teile) for label, col in TEIL_OPTIONS.items()}

    if st.button("Terminplan berechnen", type="primary", key="btn_m"):
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
        st.divider()
        show_results(_predict(row), projekt_start_m)

with tab_csv:
    st.markdown("Lade eine CSV-Datei mit Projektdaten hoch (Trennzeichen: `;` oder `,`).")
    uploaded = st.file_uploader("CSV hier ablegen oder auswählen", type=["csv"])
    if uploaded:
        try:
            raw = uploaded.read().decode("utf-8-sig")
            sep = ";" if raw.count(";") > raw.count(",") else ","
            df_up = pd.read_csv(pd.io.common.StringIO(raw), sep=sep)
            st.success(f"{len(df_up)} Projekt(e) geladen · {len(df_up.columns)} Spalten")
            st.dataframe(df_up.head(5), use_container_width=True)
            if len(df_up) > 1:
                id_col = "projekt_id" if "projekt_id" in df_up.columns else None
                if id_col:
                    sel_id  = st.selectbox("Projekt auswählen", df_up[id_col].tolist())
                    sel_row = df_up[df_up[id_col] == sel_id].iloc[0]
                else:
                    idx     = st.number_input("Zeile (0-basiert)", 0, len(df_up)-1, 0)
                    sel_row = df_up.iloc[idx]
            else:
                sel_row = df_up.iloc[0]
            ps_csv = st.date_input("Projektstart", value=date.today(), key="ps_csv")
            if st.button("Terminplan berechnen", type="primary", key="btn_csv"):
                st.divider()
                show_results(_predict(row_to_dict(sel_row)), ps_csv)
        except Exception as e:
            st.error(f"Fehler beim Lesen der CSV: {e}")
    else:
        st.info("Erwartete Spalten: `projekttyp`, `antriebsart`, `fahrzeugsegment`, … sowie alle `teil_*` Flags (0/1).")
        st.download_button(
            "Beispiel-CSV herunterladen",
            data=pd.read_csv("pep_terminplan_synthetic.csv", sep=";").head(3)
                   .to_csv(sep=";", index=False).encode("utf-8-sig"),
            file_name="pep_beispiel.csv", mime="text/csv",
        )
