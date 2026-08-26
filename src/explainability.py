from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from sklearn.pipeline import Pipeline

from src.data_loader import SIGNAL_COLS
from src.feature_engineering import FEATURE_NAMES
from src.models.lstm_classifier import DriverLSTM, DriverCNNLSTM

_LABEL_NAMES = ["Aggressive", "Normal", "Cautious"]
_COLOURS = {
    "Aggressive": "#D64045", "Normal": "#2E86AB",
    "Cautious": "#57A773",   "Not-Aggressive": "#2E86AB",
}
_PALETTE = ["#D64045", "#2E86AB", "#57A773", "#9B59B6"]


def _colour(label: str, idx: int) -> str:
    return _COLOURS.get(label, _PALETTE[idx % len(_PALETTE)])


def explain_random_forest(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    output_dir: str | Path,
    max_display: int = 15,
    shap_sample: int = 100,
    label_names: list[str] | None = None,
    positive_class_idx: int = 0,
    file_prefix: str = "random_forest",
) -> None:
    """
    SHAP via TreeExplainer run on each LODO fold model separately, importances
    averaged across folds. Consistent with the models used for evaluation.
    """
    names       = label_names or _LABEL_NAMES
    num_classes = len(names)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(42)

    all_mean_abs: list[np.ndarray] = []
    waterfall_sv: np.ndarray | None = None
    waterfall_ev: float | None = None
    waterfall_x:  np.ndarray | None = None

    rf_cls    = pipeline.named_steps["rf"].__class__
    rf_params = pipeline.named_steps["rf"].get_params()

    for held_out in sorted(np.unique(groups)):
        test_mask  = groups == held_out
        train_mask = ~test_mask

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[train_mask])

        rf = rf_cls(**rf_params)
        rf.fit(X_tr_s, y[train_mask])

        X_test_s = scaler.transform(X[test_mask])
        idx = rng.choice(len(X_test_s), size=min(shap_sample, len(X_test_s)), replace=False)
        X_s, y_s = X_test_s[idx], y[test_mask][idx]

        explainer = shap.TreeExplainer(
            rf, feature_perturbation="tree_path_dependent",
            feature_names=FEATURE_NAMES,
        )
        sv = explainer.shap_values(X_s)
        ev = np.array(explainer.expected_value)

        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        if sv.ndim == 2:
            sv = np.stack([-sv, sv], axis=-1)
            ev = np.array([1 - float(ev), float(ev)])

        all_mean_abs.append(np.abs(sv).mean(axis=0))

        if waterfall_sv is None:
            pos = np.where(y_s == positive_class_idx)[0]
            if len(pos) > 0:
                i = pos[0]
                waterfall_sv = sv[i, :, positive_class_idx]
                waterfall_ev = float(ev[positive_class_idx])
                waterfall_x  = X_s[i]

    avg = np.mean(np.stack(all_mean_abs, axis=0), axis=0)
    _plot_bar_importance(
        avg, FEATURE_NAMES, output_dir, max_display, names,
        f"shap_importance_{file_prefix}.png",
        "Random Forest — Feature Importance by Class (SHAP, avg across LODO folds)",
    )

    if waterfall_sv is not None:
        exp = shap.Explanation(
            values=waterfall_sv, base_values=waterfall_ev,
            data=waterfall_x, feature_names=FEATURE_NAMES,
        )
        shap.waterfall_plot(exp, max_display=12, show=False)
        plt.title(f"RF — {names[positive_class_idx]} prediction breakdown",
                  fontweight="bold")
        slug = names[positive_class_idx].lower().replace(" ", "_").replace("-", "_")
        out  = output_dir / f"shap_waterfall_{file_prefix}_{slug}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {out}")

    print(f"[SHAP-RF] Plots saved to {output_dir}")


def explain_lstm(
    fold_models: list[tuple],
    X: np.ndarray,
    output_dir: str | Path,
    background_samples: int = 30,
    test_samples: int = 50,
    label_names: list[str] | None = None,
    file_prefix: str = "lstm",
) -> None:
    """
    SHAP GradientExplainer on each pre-trained LODO fold model, importances
    averaged across folds. fold_models is a list of (model, scaler, test_mask)
    returned by cross_validate_lstm(return_fold_models=True) — no retraining.
    """
    names       = label_names or _LABEL_NAMES
    num_classes = len(names)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    rng    = np.random.default_rng(42)
    n, t, c = X.shape

    all_mean_abs: list[np.ndarray] = []

    for model, scaler, test_mask in fold_models:
        model.eval().to(device)
        X_test = X[test_mask]
        nt     = X_test.shape[0]
        X_s    = scaler.transform(X_test.reshape(-1, c)).reshape(nt, t, c).astype(np.float32)
        Xt     = torch.tensor(X_s, dtype=torch.float32)

        bg_idx   = rng.choice(nt, size=min(background_samples, nt), replace=False)
        test_idx = rng.choice(nt, size=min(test_samples, nt), replace=False)

        explainer = shap.GradientExplainer(model, Xt[bg_idx])
        sv_arr    = np.array(explainer.shap_values(Xt[test_idx]))
        if sv_arr.ndim == 3:
            sv_arr = sv_arr[np.newaxis]
        all_mean_abs.append(np.abs(sv_arr).mean(axis=(0, 1)))

    avg           = np.mean(np.stack(all_mean_abs, axis=0), axis=0)
    channel_names = SIGNAL_COLS[:c] if c <= len(SIGNAL_COLS) else [f"ch_{i}" for i in range(c)]

    fig, axes = plt.subplots(1, num_classes, figsize=(6 * num_classes, 6))
    if num_classes == 1:
        axes = [axes]

    for cls_idx, (ax, label) in enumerate(zip(axes, names[:num_classes])):
        importance = avg[:, cls_idx]
        ranked     = np.argsort(importance)
        ax.barh(range(c), importance[ranked], color=_colour(label, cls_idx), alpha=0.85)
        ax.set_yticks(range(c))
        ax.set_yticklabels([channel_names[i] for i in ranked], fontsize=9)
        ax.set_xlabel("Mean |SHAP| over time × samples", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold", color=_colour(label, cls_idx))
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "LSTM — Channel Importance by Class (SHAP GradientExplainer, avg across LODO folds)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = output_dir / f"shap_importance_{file_prefix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")
    print(f"[SHAP-LSTM] Plots saved to {output_dir}")


def _plot_bar_importance(
    avg_mean_abs: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    max_display: int,
    label_names: list[str],
    filename: str,
    title: str,
) -> None:
    n_classes = avg_mean_abs.shape[1]
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 7))
    if n_classes == 1:
        axes = [axes]

    for cls_idx, (ax, label) in enumerate(zip(axes, label_names[:n_classes])):
        importance = avg_mean_abs[:, cls_idx]
        ranked     = np.argsort(importance)[::-1][:max_display]
        names_r    = [feature_names[i] for i in ranked][::-1]
        values_r   = importance[ranked][::-1]

        ax.barh(range(len(names_r)), values_r, color=_colour(label, cls_idx), alpha=0.85)
        ax.set_yticks(range(len(names_r)))
        ax.set_yticklabels(names_r, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold", color=_colour(label, cls_idx))
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")
