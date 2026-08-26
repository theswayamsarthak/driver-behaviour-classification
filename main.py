"""
Driver Behaviour Classification — UAH-DriveSet
Honda DS × AIML interview project

Usage:
  python main.py --data_dir data/raw/UAH-DRIVESET-v1 --task binary
  python main.py --data_dir data/raw/UAH-DRIVESET-v1 --task multiclass
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_loader import load_dataset, windows_to_array
from src.feature_engineering import build_feature_matrix
from src.models.random_forest import train_random_forest, save_rf
from src.models.xgb_classifier import train_xgboost, save_xgb
from src.models.lstm_classifier import cross_validate_lstm, train_lstm_final, save_lstm
from src.evaluate import evaluate, evaluate_per_driver, plot_training_curves, plot_model_comparison
from src.explainability import explain_random_forest, explain_lstm

LABEL_NAMES_MULTICLASS = ["Aggressive", "Normal", "Cautious"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default="data/raw")
    p.add_argument("--task",          choices=["multiclass", "binary"], default="multiclass")
    p.add_argument("--model",         choices=["rf", "xgb", "lstm", "both", "all"],
                   default="all")
    p.add_argument("--window_sec",    type=float, default=5.0)
    p.add_argument("--overlap",       type=float, default=0.5)
    p.add_argument("--arch",          choices=["lstm", "cnn_lstm"], default="lstm")
    p.add_argument("--lstm_epochs",   type=int,   default=60)
    p.add_argument("--lstm_lr",       type=float, default=1e-3)
    p.add_argument("--lstm_hidden",   type=int,   default=128)
    p.add_argument("--lstm_layers",   type=int,   default=2)
    p.add_argument("--conv_channels", type=int,   default=64)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--focal_loss",    action="store_true", default=False)
    p.add_argument("--no_focal_loss", dest="focal_loss", action="store_false")
    p.add_argument("--focal_gamma",   type=float, default=1.0)
    p.add_argument("--no_semantic",   action="store_true")
    p.add_argument("--skip_shap",     action="store_true")
    p.add_argument("--output_dir",    default="outputs")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    plots_dir   = Path(args.output_dir) / "plots"
    models_dir  = Path(args.output_dir) / "models"
    metrics_dir = Path(args.output_dir) / "metrics"
    for d in [plots_dir, models_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[Data] Loading UAH-DriveSet...")
    windows, labels, drivers = load_dataset(
        args.data_dir,
        window_sec          = args.window_sec,
        overlap             = args.overlap,
        use_semantic_labels = not args.no_semantic,
    )
    y_raw  = np.array(labels, dtype=np.int64)
    groups = np.array(drivers)

    if args.task == "binary":
        y = (y_raw == 0).astype(np.int64)
        label_names  = ["Not-Aggressive", "Aggressive"]
        positive_idx = 1
        print("[Data] Binary task: Aggressive vs Not-Aggressive")
    else:
        y = y_raw
        label_names  = LABEL_NAMES_MULTICLASS
        positive_idx = 0

    counts = {label_names[int(k)]: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"[Data] {len(windows)} windows | {counts}")

    n_splits    = len(np.unique(groups))
    num_classes = len(label_names)
    arch_label  = args.arch.upper().replace("_", "-")
    if args.task == "binary":
        arch_label += "-Binary"

    run_rf   = args.model in ("rf",  "both", "all")
    run_xgb  = args.model in ("xgb", "all")
    run_lstm = args.model in ("lstm", "both", "all")

    all_results:   dict[str, dict]       = {}
    oof_probs_all: dict[str, np.ndarray] = {}
    X_feat: np.ndarray | None = None

    # ── Random Forest ──────────────────────────────────────────────────────────
    if run_rf:
        print("\n[RF] Engineering features...")
        X_feat = build_feature_matrix(windows)
        print("[RF] Nested LODO training (inner GridSearchCV per fold)...")
        rf_pipeline, oof_preds, rf_probs = train_random_forest(
            X_feat, y, groups, n_splits=n_splits
        )
        rf_name   = "Random Forest" + (" Binary" if args.task == "binary" else "")
        rf_prefix = "random_forest" + ("_binary" if args.task == "binary" else "")
        save_rf(rf_pipeline, models_dir / f"random_forest_{args.task}.pkl")
        all_results[rf_name] = evaluate(y, oof_preds, rf_name, plots_dir, label_names)
        evaluate_per_driver(y, oof_preds, groups, rf_name, plots_dir, label_names)
        oof_probs_all[rf_name] = rf_probs

        if not args.skip_shap:
            print("\n[SHAP] RF — computing across LODO fold models (no retraining)...")
            explain_random_forest(
                rf_pipeline, X_feat, y, groups, plots_dir,
                label_names=label_names, positive_class_idx=positive_idx,
                file_prefix=rf_prefix,
            )

    # ── XGBoost ────────────────────────────────────────────────────────────────
    if run_xgb:
        if X_feat is None:
            print("\n[XGB] Engineering features...")
            X_feat = build_feature_matrix(windows)
        print("\n[XGB] Nested LODO training (inner GridSearchCV per fold)...")
        xgb_pipeline, oof_preds, xgb_probs = train_xgboost(
            X_feat, y, groups, n_splits=n_splits
        )
        xgb_name   = "XGBoost" + (" Binary" if args.task == "binary" else "")
        save_xgb(xgb_pipeline, models_dir / f"xgboost_{args.task}.pkl")
        all_results[xgb_name] = evaluate(y, oof_preds, xgb_name, plots_dir, label_names)
        evaluate_per_driver(y, oof_preds, groups, xgb_name, plots_dir, label_names)
        oof_probs_all[xgb_name] = xgb_probs

    # ── LSTM ───────────────────────────────────────────────────────────────────
    if run_lstm:
        print(f"\n[{arch_label}] Building sequence array...")
        X_seq = windows_to_array(windows)

        lstm_kwargs = dict(
            num_classes    = num_classes,
            arch           = args.arch,
            hidden_size    = args.lstm_hidden,
            num_layers     = args.lstm_layers,
            conv_channels  = args.conv_channels,
            epochs         = args.lstm_epochs,
            lr             = args.lstm_lr,
            batch_size     = args.batch_size,
            use_focal_loss = args.focal_loss,
            focal_gamma    = args.focal_gamma,
            seed           = args.seed,
        )

        need_shap  = not args.skip_shap
        print(f"\n[{arch_label}] Leave-one-driver-out CV ({n_splits} folds)...")
        cv_result = cross_validate_lstm(
            X_seq, y, groups, return_fold_models=need_shap, **lstm_kwargs
        )
        if need_shap:
            oof_preds, lstm_probs, fold_histories, fold_models = cv_result
        else:
            oof_preds, lstm_probs, fold_histories = cv_result

        all_results[arch_label] = evaluate(y, oof_preds, arch_label, plots_dir, label_names)
        evaluate_per_driver(y, oof_preds, groups, arch_label, plots_dir, label_names)
        oof_probs_all[arch_label] = lstm_probs

        with open(metrics_dir / f"lstm_cv_fold_losses_{args.task}.json", "w") as f:
            json.dump(fold_histories, f, indent=2)

        # Final deployment model (all drivers) — used for deployment only,
        # never contributes to reported metrics
        print(f"\n[{arch_label}] Training final deployment model...")
        final_model, final_scaler, final_history = train_lstm_final(
            X_seq, y, groups, **lstm_kwargs
        )
        save_lstm(final_model, final_scaler, models_dir / f"lstm_{args.task}.pt")
        plot_training_curves(final_history, arch_label, plots_dir)

        if need_shap:
            print(f"\n[SHAP] LSTM — using fold models directly (no retraining)...")
            lstm_prefix = "lstm" + ("_binary" if args.task == "binary" else "")
            explain_lstm(
                fold_models, X_seq, plots_dir,
                label_names=label_names, file_prefix=lstm_prefix,
            )

    # ── Ensemble ───────────────────────────────────────────────────────────────
    if len(oof_probs_all) >= 2:
        stacked   = np.mean(list(oof_probs_all.values()), axis=0)
        ens_preds = stacked.argmax(axis=1)

        # Build a short readable name from model keys without truncating mid-word
        _abbrev = {"Random Forest Binary": "RF", "Random Forest": "RF",
                   "XGBoost Binary": "XGB", "XGBoost": "XGB"}
        parts    = [_abbrev.get(k, k.split()[0]) for k in oof_probs_all]
        ens_name = f"Ensemble ({'+'.join(parts)})"
        print(f"\n[Ensemble] {ens_name} — probability averaging...")
        all_results[ens_name] = evaluate(y, ens_preds, ens_name, plots_dir, label_names)
        evaluate_per_driver(y, ens_preds, groups, ens_name, plots_dir, label_names)
        np.savez(
            metrics_dir / f"ensemble_oof_probs_{args.task}.npz",
            **{k.replace(" ", "_"): v for k, v in oof_probs_all.items()},
            ensemble_probs=stacked, y_true=y, groups=groups,
        )

    if len(all_results) > 1:
        plot_model_comparison(
            all_results, plots_dir, filename=f"model_comparison_{args.task}.png"
        )

    with open(metrics_dir / f"results_{args.task}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Done] Plots   → {plots_dir}/")
    print(f"[Done] Metrics → {metrics_dir}/results_{args.task}.json")


if __name__ == "__main__":
    main()
