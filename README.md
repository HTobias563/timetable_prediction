# Timetable Prediction — PEP Milestone Scheduling

Synthetic proof-of-concept for milestone scheduling using cascading modeling.

Built as part of a bachelor's thesis on predicting initial project schedules for the
Product Development Process (PEP) in the automotive industry.

---

## Concept

The core idea is a **cascading model chain**: starting from project parameters known at
day zero, each model predicts the duration of one project phase and passes its output
as an additional feature to the next model.

```
Input Features (T0)
        │
        ▼
   Model 1  (Multi-Output)
        → dauer_start_kf_d        ← duration: project start → concept approval
        → stueckzahl_kf_refined   ← auxiliary: refined unit volume after concept approval
        │
        ▼
   Model 2  → dauer_kf_df_d       ← concept approval → design release
        │
        ▼
   Model 3  → dauer_df_tf_d       ← design release → technical release
        │
       ...
        ▼
   Model 7  → dauer_vs_sop_d      ← pre-series → start of production
```

Each downstream model receives all input features **plus** the predicted outputs of
all previous models, so uncertainty compounds naturally along the chain.

---

## Repository Structure

```
├── generate_pep_dataset.py      # Generates the synthetic master CSV
├── train_cascading_models.py    # Trains and evaluates the cascading model chain
└── pep_terminplan_synthetic.csv # Generated dataset (80 rows, 38 columns)
```

---

## Dataset

The synthetic dataset contains **80 projects** with **38 columns**.

### Input Features (known at project start)

| Category | Features |
|---|---|
| Project type | `projekttyp` (Neuanlauf / Facelift / Derivat) |
| Vehicle | `fahrzeugsegment`, `antriebsart`, `zielmarkt`, `entwicklungsstandort` |
| Planning | `geplante_stueckzahl`, `anzahl_varianten`, `aehnlichkeit_vorgaenger_pct`, `ressourcen_fte` |
| Component scope flags | `teil_verbrenner_motor`, `teil_e_motor`, `teil_getriebe`, `teil_abgasanlage`, `teil_batteriesystem`, `teil_vorderachse`, `teil_hinterachse`, `teil_lenkung`, `teil_rohbau`, `teil_aussendesign`, `teil_bordnetz`, `teil_steuergeraete_ecu`, `teil_ladesystem`, `teil_cockpit_hmi`, `teil_sitzsystem`, `teil_kamera_radar`, `teil_adas_software`, `teil_kuehlkreislauf` |
| Derived | `anzahl_teile_neu` |

### Target Variables (milestone durations in days)

| Column | Phase |
|---|---|
| `dauer_start_kf_d` | Project start → Concept approval |
| `dauer_kf_df_d` | Concept approval → Design release |
| `dauer_df_tf_d` | Design release → Technical release |
| `dauer_tf_p1_d` | Technical release → Prototype 1 |
| `dauer_p1_p2_d` | Prototype 1 → Prototype 2 |
| `dauer_p2_vs_d` | Prototype 2 → Pre-series |
| `dauer_vs_sop_d` | Pre-series → Start of production |

### Auxiliary Target

| Column | Description |
|---|---|
| `stueckzahl_kf_refined` | Refined unit volume after concept approval (predicted by Model 1 alongside the first duration) |

---

## Model

- **Algorithm:** Random Forest Regressor (`sklearn`)
- **Model 1:** `MultiOutputRegressor` — predicts `dauer_start_kf_d` and `stueckzahl_kf_refined` simultaneously
- **Models 2–7:** Single-output RF with a growing feature matrix
- **Training:** Teacher forcing — each model is trained with the true values of all previous targets
- **Evaluation:** Cascaded predictions — predicted values are passed forward, simulating real deployment

---

## Setup

```bash
pip install pandas numpy scikit-learn
python generate_pep_dataset.py   # regenerate the CSV
python train_cascading_models.py # train and evaluate the chain
```

---

## Context

This POC uses fully synthetic data to demonstrate the feasibility of the cascading
modeling approach before applying it to real project data. The synthetic data encodes
domain-realistic dependencies, e.g. new drivetrain components extend prototype phases,
new E/E architecture extends technical integration, and Neuanlauf projects take
significantly longer than Facelift or Derivat projects.
