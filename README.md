# Driver Behaviour Classification
**UAH-DriveSet | Random Forest + XGBoost + BiLSTM + Ensemble | Honda DS × AIML**

---

## Overview

Built a pipeline to classify driver behaviour from raw IMU and GPS signals,
evaluated under strict nested leave-one-driver-out cross-validation.
Two task framings:

- **Binary** — Aggressive vs Not-Aggressive (the ADAS safety decision)
- **3-class** — Aggressive / Normal / Cautious (full granularity)

Three models (RF, XGBoost, BiLSTM) plus a probability-averaged ensemble.

---

## Dataset

[UAH-DriveSet](http://www.robesafe.uah.es/personal/eduardo.romera/uahdriving/) —
6 drivers, 3 behaviour labels, 2 road types (secondary / motorway).

- `RAW_ACCELEROMETERS.txt` — IMU @ 10 Hz
- `RAW_GPS.txt` — speed @ 1 Hz, interpolated to 10 Hz
- Label from session folder name (e.g. `...-AGGRESSIVE-MOTORWAY`)

`SEMANTIC_FINAL.txt` per-second labels were attempted but couldn't be parsed
reliably — all results use session-level folder labels (standard for this dataset).

**Download:**
```bash
wget http://www.robesafe.uah.es/personal/eduardo.romera/uahdriving/UAH-DRIVESET-v1.zip
unzip UAH-DRIVESET-v1.zip -d data/raw/
python diagnose.py --data_dir data/raw
```

---

## Signal Pipeline

```
RAW_ACCELEROMETERS.txt (10 Hz) + RAW_GPS.txt (1 Hz, interpolated)
                    │
                    ▼
     7 channels: speed, acc_lon, acc_lat, acc_vert, gyro_x, gyro_y, gyro_z
                    │
                    ▼
        Sliding window — 5 s, 50% overlap (50 samples/window)
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    27 statistical        Raw (50 × 7)
    features              sequence
    (mean, std, jerk,         │
     RMS, event counts)       ▼
          │               BiLSTM (2-layer, 128 hidden)
          ├── Random Forest    also: CNN-LSTM (--arch cnn_lstm)
          └── XGBoost
```

No steering angle in this dataset — `gyro_z` (yaw rate) was the intended
steering proxy, but SHAP showed `gyro_x` (roll rate) is more discriminative.

---

## Evaluation — Nested Leave-One-Driver-Out CV

With only 6 drivers, even hyperparameter tuning can leak if not handled carefully.

**Outer loop**: hold out one driver (6 folds, one per driver).  
**Inner loop**: `GridSearchCV` on the remaining 5 drivers with their own
`GroupKFold(3)` — hyperparameters are selected without ever seeing the
held-out driver's data.

This is applied to all three models. The LSTM uses fixed-epoch training per
fold (no inner val split — 1-driver validation is too noisy to tune against).

**SHAP** is computed on each of the 6 fold models separately and averaged —
consistent with the models that produced the evaluation numbers. The fold
models are saved in memory during CV and reused directly; no retraining.

**Ensemble**: OOF probabilities averaged across all trained models, then
argmax — inherits the LODO guarantee from each component.

---

## Results

### Binary — Aggressive vs Not-Aggressive

| Model | Macro F1 (pooled OOF) | Accuracy | Aggressive Precision | Aggressive Recall | Aggressive F1 |
|---|---|---|---|---|---|
| Random Forest | 0.628 | 0.713 | 0.442 | 0.457 | 0.449 |
| XGBoost | 0.626 | **0.753** | **0.526** | 0.333 | 0.408 |
| BiLSTM (30 epochs) | 0.610 | 0.666 | 0.393 | **0.558** | 0.461 |
| **Ensemble (RF+XGB+LSTM)** | **0.647** | 0.736 | 0.483 | 0.457 | 0.470 |
| Majority baseline | ~0.427 | — | — | 0.000 | — |

**XGBoost does not beat RF on macro F1** despite higher accuracy. XGBoost
achieves accuracy by predicting Not-Aggressive heavily (recall 0.897 on
majority class) at the cost of Aggressive recall collapsing to 0.333 — the
exact wrong tradeoff for a safety system where missing aggressive behaviour
is costlier than a false alert. RF's `class_weight='balanced'` keeps it
more honest on the minority class. The ensemble partially corrects for this
by mixing in LSTM's high Aggressive recall (0.558).

**LSTM at 30 epochs is not converged** — fold losses drop 29–33% in the
second half of training, meaning the model is still learning at full speed
at epoch 30. Running to 60 epochs will improve LSTM and lift the ensemble
further. See Future Work.

### 3-class — Aggressive / Normal / Cautious

| Model | Macro F1 | Std (per-driver) |
|---|---|---|
| Random Forest | 0.470 | 0.068 |
| BiLSTM | 0.456 | 0.052 |

Random baseline ~0.333. Cautious (drowsy) vs Normal is the hard boundary —
drowsy driving doesn't produce strong IMU perturbations in a 5-second window.

### Per-driver consistency

D1 is consistently the hardest driver (lowest or near-lowest in every
configuration). D1 and D4 are the two weakest across all settings, pointing
to a genuine driver-style gap rather than a modelling issue — a production
system would need per-driver calibration rather than one global threshold.

---

## SHAP Explainability

- **RF / XGBoost**: `TreeExplainer` with `tree_path_dependent`, run per fold, averaged
- **LSTM**: `GradientExplainer` on raw sequences per fold, averaged across time

**Key finding**: `gyro_x` (roll rate) dominates feature importance across both
models and both tasks — not `gyro_z` (yaw rate, the intended steering proxy).
Hard cornering and aggressive lane changes appear to manifest more strongly
in vehicle body roll than in yaw rate itself. `acc_lat` (lateral g-force)
is second in every configuration.

Waterfall plots show the local picture can differ: for individual Aggressive
predictions, `speed_mean` and `speed_max` are the top positive contributors,
while `gyro_x_std` (the most important feature globally) sometimes pulls the
same prediction slightly negative. This is normal SHAP behaviour — global
importance averages many instances; local importance explains one.

---

## What I tried and what didn't work

**CNN-LSTM** (`--arch cnn_lstm`): Conv1d → BiLSTM. Underperforms plain LSTM
with 5 drivers per training fold — the convolutional parameters overfit.
Kept as a working option; likely worthwhile with more drivers.

**Focal loss + class weights together**: collapsed Cautious/Aggressive recall
by over-correcting for class imbalance. Fixed by making them mutually
exclusive.

**Early stopping on validation F1**: validation set = 1 driver = too noisy
to stop on reliably. Switched to fixed-epoch training with proper LODO.

**XGBoost not beating RF on macro F1**: XGBoost achieves higher accuracy
by predicting the majority class more aggressively, which hurts minority-class
recall. RF with balanced class weights is more appropriate for this
imbalanced safety task.

---

## Future Work

1. **LSTM at 60 epochs** — fold losses still declining ~30% at epoch 30;
   more epochs will improve LSTM and the ensemble
2. **More drivers** — biggest lever; 6 drivers is a hard statistical ceiling
3. **`SEMANTIC_FINAL.txt` parsing** — per-second labels for cleaner ground truth
4. **Window size ablation** — only 5 s tested
5. **Calibrated ensemble** — Platt scaling before probability averaging
6. **Proper LSTM hyperparameter search** — RF and XGBoost got nested CV with
   grid search; LSTM hyperparameters were hand-tuned

---

## Quickstart

```bash
pip install -r requirements.txt

# All three models + ensemble, binary task (recommended)
python main.py --data_dir data/raw/UAH-DRIVESET-v1 --task binary --model all

# Skip SHAP while iterating (much faster)
python main.py --data_dir data/raw/UAH-DRIVESET-v1 --task binary --model all --skip_shap

# Individual models
python main.py --data_dir data/raw/UAH-DRIVESET-v1 --task binary --model rf
python main.py --data_dir data/raw/UAH-DRIVESET-v1 --task binary --model xgb
python main.py --data_dir data/raw/UAH-DRIVESET-v1 --task binary --model lstm
```

### Key flags

| Flag | Default | Notes |
|---|---|---|
| `--task` | `multiclass` | `binary` or `multiclass` |
| `--model` | `all` | `rf`, `xgb`, `lstm`, `both` (rf+lstm), or `all` |
| `--arch` | `lstm` | `lstm` or `cnn_lstm` |
| `--lstm_epochs` | `60` | Per fold (runs 6×). Convergence warning prints if still declining |
| `--skip_shap` | off | Skip SHAP — useful while iterating |
| `--focal_loss` | off | Do not combine with class weights |
| `--no_semantic` | off | Use folder-level labels only |

---

## Project Structure

```
driver-behaviour-classification/
├── main.py
├── diagnose.py
├── recover_outputs.py
├── requirements.txt
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── models/
│   │   ├── random_forest.py     # nested LODO + inner GridSearchCV
│   │   ├── xgb_classifier.py    # same methodology, XGBoost
│   │   └── lstm_classifier.py   # LODO CV, convergence check, save/load
│   ├── evaluate.py
│   └── explainability.py        # SHAP averaged across fold models
└── outputs/
    ├── models/     # random_forest_<task>.pkl, xgboost_<task>.pkl,
    │               # lstm_<task>.pt + .scaler.pkl
    ├── plots/      # confusion matrices, per-driver charts, SHAP, training curves
    └── metrics/    # results_<task>.json, lstm_cv_fold_losses_<task>.json,
                    # ensemble_oof_probs_<task>.npz
```

---

## Citation

```bibtex
@inproceedings{romera2016need,
  title={Need data for driver behaviour analysis? Presenting the public UAH-DriveSet},
  author={Romera, Eduardo and Bergasa, Luis M and Arroyo, Roberto},
  booktitle={IEEE ITSC 2016},
  year={2016}
}
```
