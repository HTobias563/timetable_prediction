"""Shared model training, prediction, and evaluation logic."""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

DURATION_COLS = [
    "dauer_start_kf_d", "dauer_kf_df_d", "dauer_df_tf_d",
    "dauer_tf_p1_d", "dauer_p1_p2_d", "dauer_p2_vs_d", "dauer_vs_sop_d",
]
PHASE_LABELS = [
    "Projektstart → KF",
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
TEIL_COLS  = [f for f in INPUT_FEATURES if f.startswith("teil_")]

TEIL_OPTIONS = {
    "Verbrennermotor":     "teil_verbrenner_motor",
    "E-Motor":             "teil_e_motor",
    "Getriebe":            "teil_getriebe",
    "Abgasanlage":         "teil_abgasanlage",
    "Batteriesystem":      "teil_batteriesystem",
    "Vorderachse":         "teil_vorderachse",
    "Hinterachse":         "teil_hinterachse",
    "Lenkung":             "teil_lenkung",
    "Rohbau":              "teil_rohbau",
    "Außendesign":         "teil_aussendesign",
    "Bordnetz":            "teil_bordnetz",
    "Steuergeräte (ECU)":  "teil_steuergeraete_ecu",
    "Ladesystem":          "teil_ladesystem",
    "Cockpit / HMI":       "teil_cockpit_hmi",
    "Sitzsystem":          "teil_sitzsystem",
    "Kamera / Radar":      "teil_kamera_radar",
    "ADAS Software":       "teil_adas_software",
    "Kühlkreislauf":       "teil_kuehlkreislauf",
}


def _make_encoder():
    return OrdinalEncoder(
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


def _fit_cascade(X_full, y_dur, y_aux):
    """Fit all 7 models on the given feature/target matrices."""
    models = {}
    X = X_full.copy()

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

    return models


@st.cache_resource
def train_models():
    """Train on full dataset — used for the prediction page."""
    df = pd.read_csv("pep_terminplan_synthetic.csv", sep=";")
    enc = _make_encoder()
    df_enc = df.copy()
    df_enc[CAT_FEATURES] = enc.fit_transform(df[CAT_FEATURES])
    X = df_enc[INPUT_FEATURES].copy()
    y_dur = df_enc[DURATION_COLS]
    y_aux = df_enc[AUXILIARY]
    models = _fit_cascade(X, y_dur, y_aux)
    return models, enc


@st.cache_data
def evaluate_models():
    """Train/test split evaluation — used for the evaluation page."""
    df = pd.read_csv("pep_terminplan_synthetic.csv", sep=";")
    enc = _make_encoder()
    df_enc = df.copy()
    df_enc[CAT_FEATURES] = enc.fit_transform(df[CAT_FEATURES])

    idx_train, idx_test = train_test_split(df_enc.index, test_size=0.2, random_state=42)
    X_base_tr = df_enc[INPUT_FEATURES].loc[idx_train].reset_index(drop=True)
    X_base_te = df_enc[INPUT_FEATURES].loc[idx_test].reset_index(drop=True)
    y_dur_tr  = df_enc[DURATION_COLS].loc[idx_train].reset_index(drop=True)
    y_dur_te  = df_enc[DURATION_COLS].loc[idx_test].reset_index(drop=True)
    y_aux_tr  = df_enc[AUXILIARY].loc[idx_train].reset_index(drop=True)
    y_aux_te  = df_enc[AUXILIARY].loc[idx_test].reset_index(drop=True)

    models = _fit_cascade(X_base_tr, y_dur_tr, y_aux_tr)

    # Cascaded evaluation
    X_cas = X_base_te.copy()
    X_true = X_base_te.copy()
    results = []
    feat_importances = {}
    y_pred_all = {}
    y_true_all = {}

    for target in DURATION_COLS:
        model, feat_names, is_multi = models[target]

        if is_multi:
            p_true = model.predict(X_true[feat_names])[:, 0]
            p_cas  = model.predict(X_cas[feat_names])[:, 0]
            p_aux_cas = model.predict(X_cas[feat_names])[:, 1]
            X_cas[AUXILIARY]  = p_aux_cas
            X_true[AUXILIARY] = y_aux_te.values
            imp = None
        else:
            p_true = model.predict(X_true[feat_names])
            p_cas  = model.predict(X_cas[feat_names])
            imp = pd.Series(model.feature_importances_, index=feat_names)

        results.append({
            "Phase":             PHASE_LABELS[DURATION_COLS.index(target)],
            "MAE (echte Vorw.)": round(mean_absolute_error(y_dur_te[target], p_true), 1),
            "MAE (kask. Vorw.)": round(mean_absolute_error(y_dur_te[target], p_cas), 1),
            "RMSE (kask.)":      round(root_mean_squared_error(y_dur_te[target], p_cas), 1),
            "Ø Dauer (Tage)":    round(y_dur_te[target].mean(), 1),
        })

        y_pred_all[target] = p_cas
        y_true_all[target] = y_dur_te[target].values
        if imp is not None:
            feat_importances[target] = imp

        X_true[target] = y_dur_te[target].values
        X_cas[target]  = p_cas

    sop_true = y_dur_te[DURATION_COLS].sum(axis=1)
    sop_pred = pd.DataFrame(y_pred_all).sum(axis=1)
    sop_mae  = round(mean_absolute_error(sop_true, sop_pred), 1)
    sop_rmse = round(root_mean_squared_error(sop_true, sop_pred), 1)

    return (
        pd.DataFrame(results),
        feat_importances,
        sop_mae, sop_rmse,
        y_pred_all, y_true_all,
        len(idx_train), len(idx_test),
    )


def predict(row_dict, models, enc):
    df_in = pd.DataFrame([row_dict])
    df_in[CAT_FEATURES] = enc.transform(df_in[CAT_FEATURES])
    df_in["anzahl_teile_neu"] = df_in[TEIL_COLS].sum(axis=1)
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
    d = {k: row.get(k, 0) for k in [
        "projekttyp", "fahrzeugsegment", "antriebsart",
        "zielmarkt", "entwicklungsstandort",
        "geplante_stueckzahl", "anzahl_varianten",
        "aehnlichkeit_vorgaenger_pct", "ressourcen_fte",
        *TEIL_COLS,
    ]}
    d["anzahl_teile_neu"] = sum(int(d.get(c, 0)) for c in TEIL_COLS)
    return d
