import streamlit as st

st.title("Erkenntnisse")
st.markdown(
    "Was lässt sich aus dem synthetischen POC ableiten — und was nicht?"
)
st.divider()

# ── Was der POC zeigt ─────────────────────────────────────────────────────────

st.subheader("Was der POC zeigt")
st.success(
    "Die kaskadierende Modellkette funktioniert als Architektur: "
    "die Kette läuft stabil durch, Fehler pflanzen sich mathematisch korrekt fort, "
    "Multi-Output in Modell 1 ist umsetzbar und die Feature Importances sind interpretierbar. "
    "Der Ansatz ist damit prinzipiell vertretbar — das ist der eigentliche Zweck des POC."
)

st.divider()

# ── Was du NICHT schlussfolgern kannst ───────────────────────────────────────

st.subheader("Was du nicht schlussfolgern kannst")
st.error(
    "**Die Feature Importance im POC spiegelt wider, was beim Generieren einprogrammiert wurde — nicht die Realität.**  \n\n"
    "Beispiel: `projekttyp` dominiert in der Importance, weil in `generate_pep_dataset.py` "
    "explizit festgelegt wurde, dass Neuanläufe 1400–2000 Tage dauern und Facelifts nur 700–1200 Tage. "
    "Das Modell lernt genau diese Regel zurück. Das ist ein Zirkelschluss."
)

st.markdown("Daraus folgt: diese Aussagen sind auf Basis des POC **nicht** belegbar:")

col1, col2 = st.columns(2)
col1.warning("Allgemeine Projektdaten sind wichtiger als Bauteil-Flags")
col1.warning("X Features reichen, Y brauche ich nicht")
col2.warning("Die Granularität auf Bauteilebene bringt wenig Mehrwert")
col2.warning("Modell A ist besser als Modell B für dieses Problem")

st.divider()

# ── Feature-Hypothesen ────────────────────────────────────────────────────────

st.subheader("Feature-Hypothesen für echte Daten")
st.markdown(
    "Die Spalten des synthetischen Datensatzes sind eine **begründete Hypothese** darüber, "
    "welche Features in echten Daten relevant sein könnten. "
    "Diese Hypothesen müssen mit realen Projektdaten validiert oder verworfen werden."
)

hypothesen = {
    "Projekttyp (Neuanlauf / Facelift / Derivat)":
        "Stärkster Treiber der Gesamtprojektdauer — Neuanläufe dauern strukturell länger.",
    "Antriebsart (Elektro / Hybrid / Verbrenner)":
        "Elektro- und PHEV-Projekte haben mehr neue Komponenten → längere Phasen.",
    "Ähnlichkeit zum Vorgängerprojekt (%)":
        "Höhere Ähnlichkeit → weniger Neuentwicklung → kürzere Dauer.",
    "Ressourcen (FTE)":
        "Mehr Ressourcen könnten Phasendauern leicht verkürzen.",
    "Bauteil-Flags (teil_*)":
        "Spezifische Bauteile haben phasenspezifischen Einfluss — z.B. neuer E-Motor verlängert TF→P1.",
    "Anzahl neuer Bauteile (anzahl_teile_neu)":
        "Aggregierter Komplexitätsindikator — korreliert mit Gesamtdauer.",
}

for feature, hypothese in hypothesen.items():
    with st.expander(f"**{feature}**"):
        st.markdown(f"**Hypothese:** {hypothese}")
        st.markdown("**Status:** Noch nicht mit echten Daten validiert")

st.divider()

# ── Vom POC zu echten Daten ───────────────────────────────────────────────────

st.subheader("Vom POC zu echten Daten")

st.markdown("""
```
POC mit synthetischen Daten          ← hier stehen wir
        ↓
Architektur ist valide
        ↓
Feature-Katalog als Anforderungsliste
        ↓
Echte Daten beschaffen & aufbereiten
        ↓
Hypothesen validieren oder verwerfen
        ↓
Modell auf echten Daten trainieren
        ↓
Ergebnis der Bachelorarbeit
```
""")

st.info(
    "Der POC ist kein Ergebnis — er ist der **Beleg dass der Ansatz vertretbar ist**, "
    "bevor in die aufwändige Datenbeschaffung investiert wird. "
    "Genau das macht ihn gegenüber dem Betreuer argumentierbar."
)
