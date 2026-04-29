"""
Kaskadierende Modellkette für den PEP-Terminplan.

Kette:
  Model 1  (Multi-Output)
      Input : 17 Input-Features
      Output: [dauer_start_kf_d, stueckzahl_kf_refined]

  Model 2–7  (Single-Output, wachsende Feature-Matrix)
      Input : Input-Features + alle bisherigen Outputs
      Output: jeweilige Meilenstein-Dauer

Training : echte Vorwerte (teacher forcing)
Evaluation: vorhergesagte Vorwerte (echter Einsatz)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ── Daten laden ───────────────────────────────────────────────────────────────

df = pd.read_csv("pep_terminplan_synthetic.csv", sep=";")

INPUT_FEATURES = [
    # kategoriale Features
    "projekttyp",
    "fahrzeugsegment",
    "antriebsart",
    "zielmarkt",
    "entwicklungsstandort",
    # numerische Features
    "geplante_stueckzahl",
    "anzahl_varianten",
    "aehnlichkeit_vorgaenger_pct",
    "ressourcen_fte",
    # Bauteil-Flags (Schwerpunktumfänge)
    "teil_verbrenner_motor",
    "teil_e_motor",
    "teil_getriebe",
    "teil_abgasanlage",
    "teil_batteriesystem",
    "teil_vorderachse",
    "teil_hinterachse",
    "teil_lenkung",
    "teil_rohbau",
    "teil_aussendesign",
    "teil_bordnetz",
    "teil_steuergeraete_ecu",
    "teil_ladesystem",
    "teil_cockpit_hmi",
    "teil_sitzsystem",
    "teil_kamera_radar",
    "teil_adas_software",
    "teil_kuehlkreislauf",
    "anzahl_teile_neu",
]

CAT_FEATURES = [
    "projekttyp",
    "fahrzeugsegment",
    "antriebsart",
    "zielmarkt",
    "entwicklungsstandort",
]

# Auxiliary Output von Model 1 — wird ab Model 2 als Feature genutzt
AUXILIARY = "stueckzahl_kf_refined"

# Meilenstein-Dauern: Target je Modell 2–7
DURATION_COLS = [
    "dauer_start_kf_d",   # Target von Model 1 (zusammen mit AUXILIARY)
    "dauer_kf_df_d",
    "dauer_df_tf_d",
    "dauer_tf_p1_d",
    "dauer_p1_p2_d",
    "dauer_p2_vs_d",
    "dauer_vs_sop_d",
]

# ── Kategoriale Features kodieren ─────────────────────────────────────────────

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

# ── Train/Test-Split ──────────────────────────────────────────────────────────

idx_train, idx_test = train_test_split(df_enc.index, test_size=0.2, random_state=42)

X_base_train = df_enc[INPUT_FEATURES].loc[idx_train].reset_index(drop=True)
X_base_test  = df_enc[INPUT_FEATURES].loc[idx_test].reset_index(drop=True)

y_dur_train  = df_enc[DURATION_COLS].loc[idx_train].reset_index(drop=True)
y_dur_test   = df_enc[DURATION_COLS].loc[idx_test].reset_index(drop=True)

y_aux_train  = df_enc[AUXILIARY].loc[idx_train].reset_index(drop=True)
y_aux_test   = df_enc[AUXILIARY].loc[idx_test].reset_index(drop=True)

print(f"Train: {len(X_base_train)} | Test: {len(X_base_test)}\n")

# ── Akkumulierte Feature-Matrizen ─────────────────────────────────────────────
# Wachsen nach jedem Modell-Schritt um die neuen Outputs

X_train = X_base_train.copy()
X_test_cascade = X_base_test.copy()   # mit vorhergesagten Werten (echter Einsatz)
X_test_true    = X_base_test.copy()   # mit echten Werten (obere Fehlergrenze)

results = []

# ════════════════════════════════════════════════════════════════════════════════
# Model 1 — Multi-Output: dauer_start_kf_d + stueckzahl_kf_refined
# ════════════════════════════════════════════════════════════════════════════════

feat_names_m1 = list(X_train.columns)

rf_base = RandomForestRegressor(n_estimators=200, max_depth=6,
                                min_samples_leaf=3, random_state=42)
m1 = MultiOutputRegressor(rf_base)
m1.fit(X_train, pd.concat([y_dur_train["dauer_start_kf_d"], y_aux_train], axis=1))

# Predictions
pred_m1_true    = m1.predict(X_test_true[feat_names_m1])     # shape (n, 2)
pred_m1_cascade = m1.predict(X_test_cascade[feat_names_m1])

pred_kf_true,  pred_aux_true    = pred_m1_true[:, 0],    pred_m1_true[:, 1]
pred_kf_cas,   pred_aux_cas     = pred_m1_cascade[:, 0], pred_m1_cascade[:, 1]

mae_kf  = mean_absolute_error(y_dur_test["dauer_start_kf_d"], pred_kf_cas)
rmse_kf = root_mean_squared_error(y_dur_test["dauer_start_kf_d"], pred_kf_cas)
mae_aux = mean_absolute_error(y_aux_test, pred_aux_cas)

results.append({
    "modell":   "M1 (Multi-Output)",
    "target":   "dauer_start_kf_d + stueckzahl_kf_refined",
    "mae":      round(mae_kf, 1),
    "rmse":     round(rmse_kf, 1),
    "mae_aux":  round(mae_aux, 0),
})

print(f"[Model 1 — Multi-Output]")
print(f"  dauer_start_kf_d  → MAE: {mae_kf:.1f} Tage  RMSE: {rmse_kf:.1f} Tage")
print(f"  stueckzahl_kf_ref → MAE: {mae_aux:.0f} Stück\n")

# Teacher forcing: echte Werte für nächste Trainings-Iteration
X_train["dauer_start_kf_d"]   = y_dur_train["dauer_start_kf_d"].values
X_train[AUXILIARY]             = y_aux_train.values

# Test-Matrizen mit echten bzw. kaskadierten Werten befüllen
X_test_true["dauer_start_kf_d"]    = y_dur_test["dauer_start_kf_d"].values
X_test_true[AUXILIARY]             = y_aux_test.values
X_test_cascade["dauer_start_kf_d"] = pred_kf_cas
X_test_cascade[AUXILIARY]          = pred_aux_cas

# ════════════════════════════════════════════════════════════════════════════════
# Modelle 2–7 — Single-Output, wachsende Feature-Matrix
# ════════════════════════════════════════════════════════════════════════════════

for i, target in enumerate(DURATION_COLS[1:], start=2):

    feat_names = list(X_train.columns)

    rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                               min_samples_leaf=3, random_state=42)
    rf.fit(X_train, y_dur_train[target])

    pred_true    = rf.predict(X_test_true[feat_names])
    pred_cascade = rf.predict(X_test_cascade[feat_names])

    mae_t  = mean_absolute_error(y_dur_test[target], pred_true)
    rmse_t = root_mean_squared_error(y_dur_test[target], pred_true)
    mae_c  = mean_absolute_error(y_dur_test[target], pred_cascade)
    rmse_c = root_mean_squared_error(y_dur_test[target], pred_cascade)

    results.append({
        "modell":   f"M{i}",
        "target":   target,
        "mae":      round(mae_c, 1),
        "rmse":     round(rmse_c, 1),
        "mae_aux":  "-",
    })

    print(f"[Model {i} — {target}]  Features: {len(feat_names)}")
    print(f"  MAE  echte Vorwerte: {mae_t:.1f} d  |  kask. Vorwerte: {mae_c:.1f} d")
    print(f"  RMSE echte Vorwerte: {rmse_t:.1f} d  |  kask. Vorwerte: {rmse_c:.1f} d\n")

    # Teacher forcing + Cascade befüllen
    X_train[target]             = y_dur_train[target].values
    X_test_true[target]         = y_dur_test[target].values
    X_test_cascade[target]      = pred_cascade

# ── Zusammenfassung ───────────────────────────────────────────────────────────

print("══ Zusammenfassung ═════════════════════════════════════════════════")
print(pd.DataFrame(results).to_string(index=False))

# ── Gesamt-SOP-Fehler ─────────────────────────────────────────────────────────

sop_true = y_dur_test[DURATION_COLS].sum(axis=1)
sop_pred = X_test_cascade[DURATION_COLS].sum(axis=1)
print(f"\n══ Gesamt-SOP (Summe aller 7 Dauern) ══════════════════════════════")
print(f"  MAE : {mean_absolute_error(sop_true, sop_pred):.1f} Tage")
print(f"  RMSE: {root_mean_squared_error(sop_true, sop_pred):.1f} Tage")

# ── Feature Importance letztes Modell ─────────────────────────────────────────

print("\n══ Feature Importance — letztes Modell (dauer_vs_sop_d) ═══════════")
last_feat_names = [c for c in X_train.columns if c != "dauer_vs_sop_d"]
rf_last = RandomForestRegressor(n_estimators=200, max_depth=6,
                                min_samples_leaf=3, random_state=42)
rf_last.fit(X_train[last_feat_names], y_dur_train["dauer_vs_sop_d"])
imp = pd.Series(rf_last.feature_importances_, index=last_feat_names)
print(imp.sort_values(ascending=False).round(3).to_string())
