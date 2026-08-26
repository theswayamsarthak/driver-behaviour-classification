from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

LABEL_NAMES = ["Aggressive", "Normal", "Cautious"]


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str | Path,
    label_names: list[str] | None = None,
) -> dict:
    names      = label_names or LABEL_NAMES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report   = classification_report(y_true, y_pred, target_names=names,
                                     output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=names, zero_division=0))

    _plot_confusion_matrix(y_true, y_pred, model_name, output_dir, names)

    return {
        "accuracy":  float(report["accuracy"]),
        "macro_f1":  float(macro_f1),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall":    float(report[name]["recall"]),
                "f1":        float(report[name]["f1-score"]),
            }
            for name in names
        },
    }


def evaluate_per_driver(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    output_dir: str | Path,
    label_names: list[str] | None = None,
) -> dict:
    names      = label_names or LABEL_NAMES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_drivers = sorted(np.unique(groups))
    per_driver: dict[str, dict] = {}

    col_header = "  ".join(f"{n:>12}" for n in names)
    print(f"\n{'='*50}")
    print(f"  {model_name} — Per-Driver Variance")
    print(f"{'='*50}")
    print(f"  {'Driver':<10} {'Macro F1':>10}  {col_header}  N")
    print(f"  {'-'*(34 + 14 * len(names))}")

    for driver in unique_drivers:
        mask = groups == driver
        mf1  = float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))
        rep  = classification_report(y_true[mask], y_pred[mask], target_names=names,
                                     output_dict=True, zero_division=0)
        per_driver[driver] = {
            "macro_f1":  mf1,
            "n_windows": int(mask.sum()),
            "per_class": {n: float(rep[n]["f1-score"]) for n in names},
        }
        row = "  ".join(f"{per_driver[driver]['per_class'][n]:>12.3f}" for n in names)
        print(f"  {driver:<10} {mf1:>10.3f}  {row}  {mask.sum()}")

    f1_vals = [v["macro_f1"] for v in per_driver.values()]
    print(f"  {'-'*(34 + 14 * len(names))}")
    print(f"  {'Mean':<10} {np.mean(f1_vals):>10.3f}")
    print(f"  {'Std':<10} {np.std(f1_vals):>10.3f}")
    print(f"  {'Min':<10} {np.min(f1_vals):>10.3f}")
    print(f"  {'Max':<10} {np.max(f1_vals):>10.3f}")

    _plot_per_driver(per_driver, model_name, output_dir, names)
    return per_driver


def plot_training_curves(
    history: dict,
    model_name: str,
    output_dir: str | Path,
) -> None:
    if not history.get("train_loss"):
        return

    has_val = bool(history.get("val_f1"))
    epochs  = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2 if has_val else 1,
                             figsize=(12 if has_val else 6.5, 4.5))
    if not has_val:
        axes = [axes]

    fig.suptitle(f"{model_name} — Training Curves", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], color="#2E86AB", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss (final deployment model)")
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[0].grid(alpha=0.3)

    if has_val:
        best_f1 = max(history["val_f1"])
        axes[1].plot(epochs, history["val_f1"], color="#D64045", linewidth=1.2,
                     alpha=0.4, label="Raw val F1")
        if history.get("val_f1_smooth"):
            axes[1].plot(epochs, history["val_f1_smooth"], color="#D64045",
                         linewidth=2.2, label="Smoothed val F1")
        axes[1].axhline(best_f1, linestyle="--", color="grey", linewidth=1,
                        label=f"Best {best_f1:.3f}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro F1")
        axes[1].set_title("Validation Macro F1")
        axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    out  = Path(output_dir) / f"training_curves_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Training curves saved → {out}")


def plot_model_comparison(
    results: dict[str, dict],
    output_dir: str | Path,
    filename: str = "model_comparison.png",
) -> None:
    models   = list(results.keys())
    f1_vals  = [results[m]["macro_f1"] for m in models]
    acc_vals = [results[m]["accuracy"] for m in models]
    x, w     = np.arange(len(models)), 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.5), 5))
    b1 = ax.bar(x - w / 2, acc_vals, w, label="Accuracy", color="#2E86AB", alpha=0.85)
    b2 = ax.bar(x + w / 2, f1_vals,  w, label="Macro F1",  color="#D64045", alpha=0.85)

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison — Accuracy & Macro F1", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Model comparison saved → {out}")


def _plot_per_driver(
    per_driver: dict,
    model_name: str,
    output_dir: Path,
    label_names: list[str],
) -> None:
    drivers   = list(per_driver.keys())
    macro_f1s = [per_driver[d]["macro_f1"] for d in drivers]
    x         = np.arange(len(drivers))
    random_baseline = 1.0 / len(label_names)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(x, macro_f1s, color="#2E86AB", alpha=0.85, width=0.55)
    ax.axhline(np.mean(macro_f1s), color="#D64045", linestyle="--",
               linewidth=1.5, label=f"Mean {np.mean(macro_f1s):.3f}")
    ax.axhline(random_baseline, color="grey", linestyle=":", linewidth=1,
               label=f"Random ({random_baseline:.3f})")

    for bar, val in zip(bars, macro_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(drivers)
    ax.set_xlabel("Driver (held-out)")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{model_name} — Per-Driver Macro F1 (leave-one-driver-out)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    out  = output_dir / f"per_driver_f1_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Per-driver chart saved → {out}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path,
    label_names: list[str],
) -> None:
    cm      = confusion_matrix(y_true, y_pred)
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.zeros_like(cm, dtype=float)
    np.divide(cm.astype(float), row_sum, out=cm_norm, where=row_sum != 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"{model_name} — Confusion Matrix", fontsize=14,
                 fontweight="bold", y=1.01)

    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".2f"],
        ["Raw counts", "Normalised (row = true class)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    linewidths=0.5, linecolor="white",
                    xticklabels=label_names, yticklabels=label_names,
                    ax=ax, cbar_kws={"shrink": 0.8}, vmin=0)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    out  = output_dir / f"confusion_matrix_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Confusion matrix saved → {out}")
