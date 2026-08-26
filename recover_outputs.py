"""
Recovery script — regenerates ONLY missing SHAP / comparison plots using
already-saved models. Does NOT retrain anything. Run this instead of main.py
if you already have trained models on disk but some plots got overwritten
by the filename-collision bug (fixed, but damage to old output files isn't
automatically undone).

Usage:
  python recover_outputs.py --data_dir data/raw/UAH-DRIVESET-v1

Safe to run repeatedly. Skips any task/model combo it can't find a saved
file for, and tells you exactly what it skipped and why.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_loader import load_dataset, windows_to_array
from src.feature_engineering import build_feature_matrix
from src.models.random_forest import load_rf
from src.models.lstm_classifier import load_lstm
from src.explainability import explain_random_forest, explain_lstm
from src.evaluate import plot_model_comparison

LABEL_SETS = {
    "multiclass": ["Aggressive", "Normal", "Cautious"],
    "binary":     ["Not-Aggressive", "Aggressive"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    plots_dir   = Path(args.output_dir) / "plots"
    models_dir  = Path(args.output_dir) / "models"
    metrics_dir = Path(args.output_dir) / "metrics"

    print("[Recover] Loading dataset (no training, just need windows for SHAP background)...")
    windows, labels, drivers = load_dataset(args.data_dir)
    y_raw  = np.array(labels, dtype=np.int64)
    groups = np.array(drivers)
    X_feat = build_feature_matrix(windows)
    X_seq  = windows_to_array(windows)

    for task, names in LABEL_SETS.items():
        print(f"\n{'='*50}\n  Task: {task}\n{'='*50}")
        y = (y_raw == 0).astype(np.int64) if task == "binary" else y_raw
        positive_idx = 1 if task == "binary" else 0

        # --- Random Forest ---
        rf_candidates = [
            models_dir / f"random_forest_{task}.pkl",
            models_dir / "random_forest.pkl",  # legacy filename from before --task existed
        ]
        rf_path = next((p for p in rf_candidates if p.exists()), None)
        if rf_path is None:
            print(f"[Recover] RF ({task}): no saved model found at {rf_candidates} — skipping. "
                  f"You'd need to rerun --model rf --task {task} for this one.")
        else:
            print(f"[Recover] RF ({task}): found {rf_path}, regenerating SHAP...")
            rf_pipeline = load_rf(rf_path)
            file_prefix = "random_forest" + ("_binary" if task == "binary" else "")
            explain_random_forest(rf_pipeline, X_feat, y, plots_dir,
                                   label_names=names, positive_class_idx=positive_idx,
                                   file_prefix=file_prefix)

        # --- LSTM ---
        lstm_candidates = [
            models_dir / f"lstm_{task}.pt",
            models_dir / "lstm.pt",  # legacy filename from before --task existed
        ]
        lstm_path = next((p for p in lstm_candidates if p.exists()), None)
        if lstm_path is None:
            print(f"[Recover] LSTM ({task}): no saved model found at {lstm_candidates} — skipping. "
                  f"You'd need to rerun --model lstm --task {task} for this one.")
        else:
            print(f"[Recover] LSTM ({task}): found {lstm_path}, regenerating SHAP...")
            model, scaler = load_lstm(lstm_path)
            file_prefix = "lstm" + ("_binary" if task == "binary" else "")
            explain_lstm(model, scaler, X_seq, y, plots_dir,
                        label_names=names, file_prefix=file_prefix)

        # --- Model comparison (only if we have results json for this task) ---
        results_candidates = [
            metrics_dir / f"results_{task}.json",
            metrics_dir / "results.json",  # legacy filename
        ]
        results_path = next((p for p in results_candidates if p.exists()), None)
        if results_path is None:
            print(f"[Recover] Comparison ({task}): no results json found — skipping.")
        else:
            with open(results_path) as f:
                results = json.load(f)
            if len(results) > 1:
                plot_model_comparison(results, plots_dir, filename=f"model_comparison_{task}.png")
                print(f"[Recover] Comparison ({task}): regenerated from {results_path}")

    print("\n[Recover] Done. Check outputs/plots/ for the regenerated files.")


if __name__ == "__main__":
    main()
