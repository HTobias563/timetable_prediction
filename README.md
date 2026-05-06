# PEP Milestone Prediction — Proof of Concept

Proof of concept for predicting automotive project schedules using a cascading
Random Forest model chain. Built as part of a bachelor's thesis on early-stage
milestone scheduling in the Product Development Process (PEP).

---

## Concept

The core idea is a **cascading model chain**: starting from parameters known at
project start (T0), each model predicts the duration of one development phase
and passes its output as an additional feature to the next model in the chain.

```
T0 Input Features (28 parameters)
        │
        ▼
   Model 1  (Multi-Output RF)
        → dauer_start_kf_d       project start → concept approval
        → stueckzahl_kf_refined  refined unit volume after concept approval
        │
        ▼
   Model 2  → dauer_kf_df_d      concept approval → design release
        │
        ▼
   Model 3  → dauer_df_tf_d      design release → technical release
        │
        ▼
   Model 4  → dauer_tf_p1_d      technical release → prototype 1
        │
        ▼
   Model 5  → dauer_p1_p2_d      prototype 1 → prototype 2
        │
        ▼
   Model 6  → dauer_p2_vs_d      prototype 2 → pre-series
        │
        ▼
   Model 7  → dauer_vs_sop_d     pre-series → start of production
```

Each downstream model receives all T0 input features **plus** the predicted
outputs of all previous models, so uncertainty compounds naturally along the
chain. Models are trained with teacher forcing (true predecessor values) and
evaluated with cascaded predictions (predicted predecessor values) to reflect
real deployment conditions.

---

## Repository Structure

```
├── app.py                        Streamlit entry point
├── model.py                      Shared training, prediction, and evaluation logic
├── pages/
│   ├── vorhersage.py             Prediction page (manual input + CSV upload)
│   ├── simulation.py             Scenario comparison page
│   ├── trainingsdaten.py         Dataset explorer
│   ├── evaluation.py             Model evaluation and metrics
│   ├── methodik.py               Methodology explanation (cascade + Markov)
│   └── erkenntnisse.py           Findings and real-data requirements
├── generate_pep_dataset.py       Generates the synthetic dataset
├── train_cascading_models.py     Standalone training and evaluation script
├── pep_terminplan_synthetic.csv  Synthetic dataset (80 rows, 38 columns)
└── requirements.txt
```

---

## Streamlit App

The interactive app has four pages:

| Page | Description |
|---|---|
| **Vorhersage** | Enter project parameters and receive a predicted milestone schedule |
| **Simulation** | Compare two project scenarios side by side |
| **Trainingsdaten** | Explore the synthetic dataset and feature distributions |
| **Evaluation** | Model accuracy, phase-level error metrics, and RF vs. baseline comparison |

### Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

The synthetic dataset contains **80 projects** with **38 columns**.

### Input Features — known at project start (T0)

| Category | Features |
|---|---|
| Project type | `projekttyp` (Neuanlauf / Facelift / Derivat) |
| Vehicle | `fahrzeugsegment`, `antriebsart`, `zielmarkt`, `entwicklungsstandort` |
| Planning | `geplante_stueckzahl`, `anzahl_varianten`, `aehnlichkeit_vorgaenger_pct`, `ressourcen_fte` |
| Component flags | `teil_verbrenner_motor`, `teil_e_motor`, `teil_getriebe`, `teil_abgasanlage`, `teil_batteriesystem`, `teil_vorderachse`, `teil_hinterachse`, `teil_lenkung`, `teil_rohbau`, `teil_aussendesign`, `teil_bordnetz`, `teil_steuergeraete_ecu`, `teil_ladesystem`, `teil_cockpit_hmi`, `teil_sitzsystem`, `teil_kamera_radar`, `teil_adas_software`, `teil_kuehlkreislauf` |
| Derived | `anzahl_teile_neu` |

### Target Variables — milestone durations in days

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
| `stueckzahl_kf_refined` | Refined unit volume after concept approval — predicted jointly with the first phase duration by Model 1 |

---

## Model

- **Algorithm:** Random Forest Regressor (`scikit-learn`)
- **Model 1:** `MultiOutputRegressor` — predicts `dauer_start_kf_d` and `stueckzahl_kf_refined` simultaneously
- **Models 2–7:** Single-output RF with a growing feature matrix (28 → 35 features)
- **Hyperparameters:** `n_estimators=200`, `max_depth=6`, `min_samples_leaf=3`
- **Training:** Teacher forcing — each model trains on the true predecessor values
- **Evaluation:** Cascaded predictions — predicted values are passed forward, simulating real deployment

### Baseline

A semi-Markov baseline is included for comparison: phase durations are estimated
as the historical mean per project type, without any learning algorithm. The RF
cascade is benchmarked against this baseline to quantify the value added by the
learned model.

---

## Regenerate Dataset / Retrain

```bash
python generate_pep_dataset.py   # regenerates pep_terminplan_synthetic.csv
python train_cascading_models.py # trains the chain and prints evaluation metrics
```

---

## Context

This POC uses fully synthetic data to validate the cascading modeling approach
before applying it to real project data. The synthetic dataset encodes
domain-realistic dependencies — for example, new E/E components extend
technical integration phases, and Neuanlauf projects take significantly longer
than Facelift or Derivat projects. The POC demonstrates that the architecture
is viable and motivates the real-data collection effort.
