import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model import (
    train_models, predict,
    DURATION_COLS, PHASE_LABELS, PHASE_COLORS, TEIL_OPTIONS, AUXILIARY,
)

with st.spinner("Modelle werden geladen..."):
    models, enc = train_models()

st.title("Szenario-Simulation")
st.markdown(
    "Stelle zwei Projektszenarien gegenüber — und sieh sofort, "
    "wie sich unterschiedliche Parameter auf den Terminplan auswirken."
)
st.divider()

# ── Vorlagen ──────────────────────────────────────────────────────────────────

PRESETS = {
    "Neuanlauf · Elektro": {
        "projekttyp": "Neuanlauf", "antriebsart": "Elektro",
        "fahrzeugsegment": "SUV", "zielmarkt": "Global",
        "entwicklungsstandort": "Deutschland",
        "geplante_stueckzahl": 50_000, "anzahl_varianten": 3,
        "aehnlichkeit": 15, "ressourcen_fte": 120,
        "teile": ["E-Motor", "Batteriesystem", "Ladesystem", "Rohbau",
                  "Steuergeräte (ECU)", "ADAS Software", "Bordnetz"],
    },
    "Facelift · Verbrenner": {
        "projekttyp": "Facelift", "antriebsart": "Verbrenner",
        "fahrzeugsegment": "C", "zielmarkt": "Europa",
        "entwicklungsstandort": "Deutschland",
        "geplante_stueckzahl": 80_000, "anzahl_varianten": 2,
        "aehnlichkeit": 75, "ressourcen_fte": 60,
        "teile": ["Außendesign", "Cockpit / HMI"],
    },
    "Derivat · Hybrid": {
        "projekttyp": "Derivat", "antriebsart": "Hybrid",
        "fahrzeugsegment": "B", "zielmarkt": "Europa",
        "entwicklungsstandort": "Deutschland",
        "geplante_stueckzahl": 60_000, "anzahl_varianten": 2,
        "aehnlichkeit": 55, "ressourcen_fte": 80,
        "teile": ["E-Motor", "Batteriesystem", "Steuergeräte (ECU)"],
    },
    "Neuanlauf · PHEV": {
        "projekttyp": "Neuanlauf", "antriebsart": "PHEV",
        "fahrzeugsegment": "D", "zielmarkt": "China",
        "entwicklungsstandort": "China",
        "geplante_stueckzahl": 100_000, "anzahl_varianten": 4,
        "aehnlichkeit": 20, "ressourcen_fte": 150,
        "teile": ["E-Motor", "Batteriesystem", "Verbrennermotor",
                  "Getriebe", "Rohbau", "Steuergeräte (ECU)", "Ladesystem"],
    },
}

PRESET_NAMES = list(PRESETS.keys())


# ── Session-State ─────────────────────────────────────────────────────────────

def _apply_preset(s: str, name: str):
    p = PRESETS[name]
    st.session_state[f"pt_{s}"]    = p["projekttyp"]
    st.session_state[f"at_{s}"]    = p["antriebsart"]
    st.session_state[f"ae_{s}"]    = p["aehnlichkeit"]
    st.session_state[f"fte_{s}"]   = p["ressourcen_fte"]
    st.session_state[f"teile_{s}"] = p["teile"]


def _init(s: str, default: str):
    if f"preset_{s}" not in st.session_state:
        st.session_state[f"preset_{s}"] = default
        _apply_preset(s, default)


def _on_preset(s: str):
    _apply_preset(s, st.session_state[f"preset_{s}"])


_init("A", "Neuanlauf · Elektro")
_init("B", "Facelift · Verbrenner")


# ── Formular ──────────────────────────────────────────────────────────────────

def scenario_form(col, s: str, label: str):
    with col:
        st.markdown(f"### {label}")
        st.selectbox(
            "Vorlage", PRESET_NAMES,
            index=PRESET_NAMES.index(st.session_state[f"preset_{s}"]),
            key=f"preset_{s}", on_change=_on_preset, args=(s,),
        )
        st.selectbox(
            "Projekttyp", ["Neuanlauf", "Facelift", "Derivat"],
            index=["Neuanlauf", "Facelift", "Derivat"].index(st.session_state[f"pt_{s}"]),
            key=f"pt_{s}",
        )
        st.selectbox(
            "Antriebsart", ["Verbrenner", "Hybrid", "PHEV", "Elektro"],
            index=["Verbrenner", "Hybrid", "PHEV", "Elektro"].index(st.session_state[f"at_{s}"]),
            key=f"at_{s}",
        )
        st.slider("Ähnlichkeit zum Vorgänger (%)", 0, 100, key=f"ae_{s}")
        st.slider("Ressourcen (FTE)", 20, 300, key=f"fte_{s}")
        st.multiselect("Neue Bauteile", list(TEIL_OPTIONS.keys()), key=f"teile_{s}")


def build_row(s: str) -> dict:
    p       = PRESETS[st.session_state[f"preset_{s}"]]
    selected = st.session_state[f"teile_{s}"]
    flags   = {col: int(lbl in selected) for lbl, col in TEIL_OPTIONS.items()}
    return {
        "projekttyp":               st.session_state[f"pt_{s}"],
        "antriebsart":              st.session_state[f"at_{s}"],
        "fahrzeugsegment":          p["fahrzeugsegment"],
        "zielmarkt":                p["zielmarkt"],
        "entwicklungsstandort":     p["entwicklungsstandort"],
        "geplante_stueckzahl":      p["geplante_stueckzahl"],
        "anzahl_varianten":         p["anzahl_varianten"],
        "aehnlichkeit_vorgaenger_pct": st.session_state[f"ae_{s}"],
        "ressourcen_fte":           st.session_state[f"fte_{s}"],
        "anzahl_teile_neu":         sum(flags.values()),
        **flags,
    }


col_a, col_b = st.columns(2, gap="large")
scenario_form(col_a, "A", "Szenario A")
scenario_form(col_b, "B", "Szenario B")

st.divider()

if st.button("Szenarien vergleichen", type="primary", use_container_width=True):
    with st.spinner("Terminpläne werden berechnet..."):
        st.session_state["sim_preds_a"] = predict(build_row("A"), models, enc)
        st.session_state["sim_preds_b"] = predict(build_row("B"), models, enc)
        st.session_state["sim_label_a"] = st.session_state["preset_A"]
        st.session_state["sim_label_b"] = st.session_state["preset_B"]

if "sim_preds_a" not in st.session_state:
    st.info("Konfiguriere zwei Szenarien und klicke auf **Szenarien vergleichen**.")
    st.stop()


# ── Ergebnis ──────────────────────────────────────────────────────────────────

preds_a = st.session_state["sim_preds_a"]
preds_b = st.session_state["sim_preds_b"]
lbl_a   = st.session_state["sim_label_a"]
lbl_b   = st.session_state["sim_label_b"]

dur_a = [preds_a[c] for c in DURATION_COLS]
dur_b = [preds_b[c] for c in DURATION_COLS]
cum_a = np.cumsum([0] + dur_a)
cum_b = np.cumsum([0] + dur_b)
tot_a = int(cum_a[-1])
tot_b = int(cum_b[-1])
delta = tot_a - tot_b


def _fmt(days: int) -> str:
    y, rest = divmod(days, 365)
    m = rest // 30
    return f"{y} J {m} M" if y else f"{m} Monate"


st.subheader("Ergebnis auf einen Blick")
m1, m2, m3 = st.columns(3)
m1.metric(f"Szenario A — {lbl_a}", _fmt(tot_a), f"{tot_a} Tage")
m2.metric(f"Szenario B — {lbl_b}", _fmt(tot_b), f"{tot_b} Tage")
sign = "+" if delta >= 0 else ""
m3.metric(
    "Unterschied  A − B",
    f"{sign}{delta} Tage",
    f"{sign}{delta / 365:.1f} Jahre",
    delta_color="inverse",
)

longer = "A" if delta > 0 else "B"
st.caption(
    f"Szenario **{longer}** dauert voraussichtlich **{abs(delta)} Tage "
    f"(~{abs(delta)/365:.1f} Jahre) länger** als Szenario {'B' if longer == 'A' else 'A'}."
)
st.divider()


# ── Gantt-Vergleich ───────────────────────────────────────────────────────────

st.subheader("Terminplan-Vergleich")
st.caption(
    f"Volle Deckkraft = Szenario A ({lbl_a}) · "
    f"Halbe Deckkraft = Szenario B ({lbl_b})"
)

fig = go.Figure()

for i, (label, dur, start) in enumerate(zip(PHASE_LABELS, dur_a, cum_a[:-1])):
    fig.add_trace(go.Bar(
        y=[f"A  ·  {label}"], x=[dur], base=[start],
        orientation="h", marker_color=PHASE_COLORS[i], opacity=0.9,
        text=f"{int(dur)} T", textposition="inside", insidetextanchor="middle",
        name="Szenario A", legendgroup="A", showlegend=(i == 0),
        hovertemplate=f"<b>A · {label}</b><br>{int(dur)} Tage<extra></extra>",
    ))

for i, (label, dur, start) in enumerate(zip(PHASE_LABELS, dur_b, cum_b[:-1])):
    fig.add_trace(go.Bar(
        y=[f"B  ·  {label}"], x=[dur], base=[start],
        orientation="h", marker_color=PHASE_COLORS[i], opacity=0.45,
        text=f"{int(dur)} T", textposition="inside", insidetextanchor="middle",
        name="Szenario B", legendgroup="B", showlegend=(i == 0),
        hovertemplate=f"<b>B · {label}</b><br>{int(dur)} Tage<extra></extra>",
    ))

ax_max = max(tot_a, tot_b) * 1.14
fig.add_vline(x=tot_a, line_dash="solid", line_color="#1f4e79", line_width=2.5)
fig.add_vline(x=tot_b, line_dash="dash",  line_color="#843c0c", line_width=2.5)
fig.add_annotation(
    x=tot_a, y=1.06, yref="paper",
    text=f"<b>SOP A: {tot_a} Tage ({_fmt(tot_a)})</b>",
    showarrow=False, font=dict(size=11, color="#1f4e79"), xanchor="left",
)
fig.add_annotation(
    x=tot_b, y=-0.06, yref="paper",
    text=f"<b>SOP B: {tot_b} Tage ({_fmt(tot_b)})</b>",
    showarrow=False, font=dict(size=11, color="#843c0c"), xanchor="left",
)
fig.update_layout(
    barmode="overlay", height=560,
    margin=dict(t=70, b=60, l=10, r=10),
    xaxis=dict(title="Tage ab Projektstart", range=[0, ax_max]),
    yaxis=dict(autorange="reversed"),
    plot_bgcolor="#f8f9fa",
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig, use_container_width=True)


# ── Phasentabelle ─────────────────────────────────────────────────────────────

st.subheader("Phase für Phase")
rows = []
for label, da, db in zip(PHASE_LABELS, dur_a, dur_b):
    d = int(da) - int(db)
    rows.append({"Phase": label, f"A (Tage)": int(da), f"B (Tage)": int(db), "A − B (Tage)": d})

df_cmp = pd.DataFrame(rows)
st.dataframe(
    df_cmp.style.background_gradient(
        subset=["A − B (Tage)"], cmap="RdYlGn_r", vmin=-250, vmax=250
    ),
    hide_index=True, use_container_width=True,
)
st.caption("Rot = Szenario A dauert in dieser Phase länger · Grün = Szenario A ist schneller")
