import streamlit as st

st.title("Datenanforderungen")
st.markdown(
    "Welche Daten werden benötigt, um das Modell mit echten Projektdaten zu trainieren?"
)
st.divider()

# ── Mindestumfang ─────────────────────────────────────────────────────────────

st.subheader("Mindestumfang")
c1, c2, c3 = st.columns(3)
c1.metric("Projekte (Minimum)", "~150–200",
          help="Unterhalb dieser Grenze sind Random-Forest-Modelle bei ~30 Features instabil")
c2.metric("Empfohlener Zeitraum", "8–10 Jahre",
          help="Um verschiedene Projekttypen und Antriebsarten ausreichend abzudecken")
c3.metric("Kritische Spalten", "T0-Planwerte",
          help="Nicht Ist-Daten, sondern die Werte die am Projektstart vorlagen")

st.divider()

# ── Qualitätsanforderungen ────────────────────────────────────────────────────

st.subheader("Qualitätsanforderungen je Feature")

anforderungen = [
    ("T0-Verfügbarkeit",      "Jedes Feature muss am Projektstart bekannt gewesen sein — keine nachträglichen Einträge",      "Hoch"),
    ("Konsistenz",            "Gleiche Meilenstein-Definition über alle Projekte (z.B. was ist 'Konzeptfreigabe'?)",            "Hoch"),
    ("Vollständigkeit",       "Fehlende Werte bei Kern-Features (Projekttyp, Antrieb) sind nicht tolerierbar",                 "Hoch"),
    ("Granularität",          "Bauteil-Flags idealerweise auf Baugruppenebene, nicht auf Einzelteil-Ebene",                    "Mittel"),
    ("Historische Plandaten", "Nicht der tatsächliche Terminplan, sondern der initiale Plan — dieser ist die Zielvariable",    "Hoch"),
]

for name, beschreibung, prio in anforderungen:
    col_n, col_d, col_p = st.columns([1, 3, 1])
    col_n.markdown(f"**{name}**")
    col_d.markdown(beschreibung)
    if prio == "Hoch":
        col_p.error("Hoch")
    else:
        col_p.warning("Mittel")

st.divider()

# ── Typische Datenhürden ──────────────────────────────────────────────────────

st.subheader("Typische Datenhürden in der Praxis")

h1, h2 = st.columns(2)

with h1:
    st.markdown("**Organisatorisch**")
    st.markdown("""
- Daten liegen in verschiedenen Systemen (SAP, Jira, Excel)
- Freigaben durch Betriebsrat / Datenschutz nötig
- Kein zentrales Projektregister vorhanden
- Zuständige wechseln über 8–10 Jahre
""")

with h2:
    st.markdown("**Technisch / inhaltlich**")
    st.markdown("""
- Meilenstein-Definitionen ändern sich über die Jahre
- Initiale Planwerte wurden überschrieben (nur Ist bleibt)
- Bauteil-Umfänge sind selten strukturiert dokumentiert
- Ältere Projekte haben Lücken in den Planungsdaten
""")
