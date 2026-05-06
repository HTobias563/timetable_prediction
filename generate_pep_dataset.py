"""
Synthetischer PEP-Terminplan-Datensatz für die Bachelorarbeit.

Erzeugt 80 fiktive Automotive-Projekte mit realistischen Abhängigkeiten
zwischen Projektparametern und Meilenstein-Dauern.

Targets der kaskadierenden Modellkette:

  Input-Features (T0)
      │
      ▼
  Model 1 (Multi-Output)
      → dauer_start_kf_d       Projektstart → Konzeptfreigabe
      → stueckzahl_kf_refined  Auxiliary: verfeinerte Stückzahl nach KF
      │
      ▼
  Model 2  → dauer_kf_df_d    Konzeptfreigabe → Designfreigabe
      │
      ▼
  ...  → Model 7 → dauer_vs_sop_d  Vorserie → SOP
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 80

# ── Kategoriale Features ─────────────────────────────────────────────────────

projekttyp = rng.choice(
    ["Neuanlauf", "Facelift", "Derivat"],
    size=N,
    p=[0.35, 0.40, 0.25],
)

fahrzeugsegment = rng.choice(
    ["A", "B", "C", "D", "SUV", "Van"],
    size=N,
    p=[0.10, 0.15, 0.20, 0.15, 0.30, 0.10],
)

antriebsart = rng.choice(
    ["Verbrenner", "Hybrid", "PHEV", "Elektro"],
    size=N,
    p=[0.30, 0.25, 0.20, 0.25],
)

zielmarkt = rng.choice(
    ["Europa", "USA", "China", "Global"],
    size=N,
    p=[0.30, 0.20, 0.25, 0.25],
)

entwicklungsstandort = rng.choice(
    ["Deutschland", "USA", "China", "Indien"],
    size=N,
    p=[0.45, 0.20, 0.25, 0.10],
)

# ── Numerische Basis-Features ─────────────────────────────────────────────────

geplante_stueckzahl      = rng.integers(5_000, 200_001, size=N)
anzahl_varianten         = rng.integers(1, 9, size=N)
aehnlichkeit_vorgaenger_pct = rng.integers(0, 101, size=N)
ressourcen_fte           = rng.integers(20, 301, size=N)

# ── Schwerpunktumfang-Flags: konkrete Bauteile (1 = Neuentwicklung) ───────────
#
#  Jedes Flag steht für ein spezifisches Bauteil / eine Baugruppe.
#  Wahrscheinlichkeiten sind abhängig von Projekttyp UND Antriebsart,
#  um realistische Korrelationen abzubilden.
#
#  p(flag=1) je Projekttyp:  Neuanlauf / Facelift / Derivat

def teil(p_neu, p_face, p_der, nur_wenn=None):
    """Bauteil-Flag, optional nur aktiv wenn eine Bedingung erfüllt ist."""
    p = np.where(projekttyp == "Neuanlauf", p_neu,
        np.where(projekttyp == "Facelift",  p_face, p_der))
    flag = rng.binomial(1, p, size=N)
    if nur_wenn is not None:
        flag = flag * nur_wenn   # erzwinge 0 wenn Bedingung nicht erfüllt
    return flag

hat_verbrenner = (antriebsart == "Verbrenner") | (antriebsart == "Hybrid")
hat_e_antrieb  = (antriebsart == "Elektro")   | (antriebsart == "PHEV") | (antriebsart == "Hybrid")

# Antrieb
teil_verbrenner_motor   = teil(0.85, 0.10, 0.05, nur_wenn=hat_verbrenner.astype(int))
teil_e_motor            = teil(0.90, 0.15, 0.05, nur_wenn=hat_e_antrieb.astype(int))
teil_getriebe           = teil(0.80, 0.15, 0.10)
teil_abgasanlage        = teil(0.75, 0.10, 0.05, nur_wenn=hat_verbrenner.astype(int))
teil_batteriesystem     = teil(0.90, 0.20, 0.10, nur_wenn=hat_e_antrieb.astype(int))

# Fahrwerk
teil_vorderachse        = teil(0.75, 0.15, 0.10)
teil_hinterachse        = teil(0.70, 0.10, 0.10)
teil_lenkung            = teil(0.60, 0.20, 0.10)

# Karosserie & Design
teil_rohbau             = teil(0.80, 0.20, 0.10)
teil_aussendesign       = teil(0.85, 0.75, 0.20)

# Elektrik / Elektronik
teil_bordnetz           = teil(0.75, 0.30, 0.15)
teil_steuergeraete_ecu  = teil(0.80, 0.35, 0.20)
teil_ladesystem         = teil(0.85, 0.25, 0.10, nur_wenn=hat_e_antrieb.astype(int))

# Innenraum & HMI
teil_cockpit_hmi        = teil(0.75, 0.65, 0.25)
teil_sitzsystem         = teil(0.60, 0.50, 0.15)

# ADAS
teil_kamera_radar       = teil(0.70, 0.35, 0.15)
teil_adas_software      = teil(0.65, 0.30, 0.10)

# Thermomanagement
teil_kuehlkreislauf     = teil(0.65, 0.10, 0.05)

TEIL_COLS = [
    "teil_verbrenner_motor", "teil_e_motor", "teil_getriebe",
    "teil_abgasanlage", "teil_batteriesystem",
    "teil_vorderachse", "teil_hinterachse", "teil_lenkung",
    "teil_rohbau", "teil_aussendesign",
    "teil_bordnetz", "teil_steuergeraete_ecu", "teil_ladesystem",
    "teil_cockpit_hmi", "teil_sitzsystem",
    "teil_kamera_radar", "teil_adas_software",
    "teil_kuehlkreislauf",
]

teil_arrays = [
    teil_verbrenner_motor, teil_e_motor, teil_getriebe,
    teil_abgasanlage, teil_batteriesystem,
    teil_vorderachse, teil_hinterachse, teil_lenkung,
    teil_rohbau, teil_aussendesign,
    teil_bordnetz, teil_steuergeraete_ecu, teil_ladesystem,
    teil_cockpit_hmi, teil_sitzsystem,
    teil_kamera_radar, teil_adas_software,
    teil_kuehlkreislauf,
]

anzahl_teile_neu = sum(teil_arrays)   # Gesamtanzahl neuer Bauteile je Projekt

# ── Auxiliary Target: verfeinerte Stückzahl nach Konzeptfreigabe ──────────────
#
#  Realität: geplante_stueckzahl ist am T0 ungenau (±30%).
#  Nach der KF ist die Stückzahl verbindlicher (±10%).
#  Das Modell lernt, diese Verfeinerung vorherzusagen.

true_stueckzahl = geplante_stueckzahl.copy()
stueckzahl_kf_refined = (
    true_stueckzahl * rng.uniform(0.90, 1.10, size=N)
).astype(int)

# Initiale Planung hat mehr Rauschen
geplante_stueckzahl = (
    true_stueckzahl * rng.uniform(0.70, 1.30, size=N)
).astype(int)

# ── Basiswert t_sop je Projekttyp ────────────────────────────────────────────

base_sop = np.where(
    projekttyp == "Neuanlauf",  rng.integers(1400, 2001, size=N),
    np.where(
        projekttyp == "Facelift", rng.integers(700, 1201, size=N),
        rng.integers(400, 801, size=N),
    ),
)

# ── Adjustierungen auf t_sop ──────────────────────────────────────────────────

antrieb_delta       = np.select(
    [antriebsart == "Elektro", antriebsart == "PHEV", antriebsart == "Hybrid"],
    [150, 80, 40], default=0,
)
aehnlichkeit_delta  = -(aehnlichkeit_vorgaenger_pct - 50) * 2
ressourcen_delta    = -(ressourcen_fte - 150) * 0.3
umfang_delta        = anzahl_teile_neu * 12  # jedes neue Bauteil + 12 Tage

noise = rng.normal(0, 40, size=N)

t_sop = (
    base_sop
    + antrieb_delta
    + aehnlichkeit_delta
    + ressourcen_delta
    + umfang_delta
    + noise
).astype(int)
t_sop = np.clip(t_sop, 300, 2400)

# ── Meilensteine ──────────────────────────────────────────────────────────────
#
#  Spezifische Bauteile verschieben einzelne Phasen gezielt:
#    Rohbau + Außendesign     → verlängert KF→DF  (Designfreigabe)
#    ECU + Bordnetz           → verlängert DF→TF  (Technische Integration)
#    E-Motor + Verbrenner     → verlängert TF→P1  (erster Prototyp)
#    Kamera/Radar + ADAS-SW   → verlängert P1→P2  (Validierung)
#    Kühlkreislauf            → verlängert P2→VS

def milestone(t_sop, lo, hi, prev, extra=None, noise_std=20):
    frac = rng.uniform(lo, hi, size=N)
    val = frac * t_sop + rng.normal(0, noise_std, size=N)
    if extra is not None:
        val += extra
    return np.maximum(val.astype(int), prev + 10)

t_kf = milestone(t_sop, 0.12, 0.18, np.zeros(N, int))
t_df = milestone(t_sop, 0.22, 0.28, t_kf,
                 extra=teil_rohbau * 15 + teil_aussendesign * 20)
t_tf = milestone(t_sop, 0.38, 0.45, t_df,
                 extra=teil_steuergeraete_ecu * 25 + teil_bordnetz * 15)
t_p1 = milestone(t_sop, 0.48, 0.55, t_tf,
                 extra=teil_e_motor * 30 + teil_verbrenner_motor * 20 + teil_batteriesystem * 25)
t_p2 = milestone(t_sop, 0.62, 0.70, t_p1,
                 extra=teil_kamera_radar * 20 + teil_adas_software * 20)
t_vs = milestone(t_sop, 0.80, 0.88, t_p2,
                 extra=teil_kuehlkreislauf * 15)

# ── Projekt-IDs und -Namen ────────────────────────────────────────────────────

projekt_ids  = [f"PRJ-{str(i+1).zfill(3)}" for i in range(N)]
marken       = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
modelle      = ["100", "200", "300", "400", "X1", "X2", "X3", "GT", "S"]
projekt_namen = [
    f"{rng.choice(marken)}_{rng.choice(modelle)}_{projekttyp[i][:3].upper()}"
    for i in range(N)
]

# ── DataFrame zusammenbauen ───────────────────────────────────────────────────

df = pd.DataFrame({
    # Identifikatoren
    "projekt_id":                    projekt_ids,
    "projekt_name":                  projekt_namen,

    # Input-Features (T0 — alle am Projektstart bekannt)
    "projekttyp":                    projekttyp,
    "fahrzeugsegment":               fahrzeugsegment,
    "antriebsart":                   antriebsart,
    "zielmarkt":                     zielmarkt,
    "entwicklungsstandort":          entwicklungsstandort,
    "geplante_stueckzahl":           geplante_stueckzahl,
    "anzahl_varianten":              anzahl_varianten,
    "aehnlichkeit_vorgaenger_pct":   aehnlichkeit_vorgaenger_pct,
    "ressourcen_fte":                ressourcen_fte,

    # Schwerpunktumfang-Flags: konkrete Bauteile (binär, T0 bekannt)
    **dict(zip(TEIL_COLS, teil_arrays)),
    "anzahl_teile_neu":              anzahl_teile_neu,

    # Auxiliary Target von Model 1 (verfeinerte Info nach KF)
    "stueckzahl_kf_refined":         stueckzahl_kf_refined,

    # Meilenstein-Dauern (Ziel-Variablen der kaskadierenden Kette)
    "dauer_start_kf_d":  t_kf,
    "dauer_kf_df_d":     t_df - t_kf,
    "dauer_df_tf_d":     t_tf - t_df,
    "dauer_tf_p1_d":     t_p1 - t_tf,
    "dauer_p1_p2_d":     t_p2 - t_p1,
    "dauer_p2_vs_d":     t_vs - t_p2,
    "dauer_vs_sop_d":    t_sop - t_vs,
})

# ── Speichern ─────────────────────────────────────────────────────────────────

out_csv = "pep_terminplan_synthetic.csv"
df.to_csv(out_csv, index=False, sep=";", encoding="utf-8-sig")

# ── Statistik ─────────────────────────────────────────────────────────────────

dauer_cols = [
    "dauer_start_kf_d", "dauer_kf_df_d", "dauer_df_tf_d",
    "dauer_tf_p1_d", "dauer_p1_p2_d", "dauer_p2_vs_d", "dauer_vs_sop_d",
]

print(f"Datensatz: {out_csv}  ({len(df)} Zeilen, {len(df.columns)} Spalten)\n")

print("── Verteilung Projekttyp ──")
print(df["projekttyp"].value_counts().to_string())

print("\n── Bauteile (Anteil Projekte mit Neuentwicklung=1) ──")
print(df[TEIL_COLS].mean().round(2).to_string())
print(f"\n  Ø Anzahl neuer Bauteile je Projekt: {df['anzahl_teile_neu'].mean():.1f}")

print("\n── Auxiliary: stueckzahl_kf_refined vs geplante_stueckzahl ──")
diff = (df["stueckzahl_kf_refined"] - df["geplante_stueckzahl"]).abs()
print(f"  Mittlere Abweichung: {diff.mean():.0f} Stück  |  Max: {diff.max():.0f}")

print("\n── Dauer-Statistik (Tage je Abschnitt) ──")
print(df[dauer_cols].describe().round(0).to_string())

print("\n── Positivitäts-Check ──")
for col in dauer_cols:
    ok = (df[col] > 0).all()
    print(f"  {col}: {'OK' if ok else 'FEHLER'}")
