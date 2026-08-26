from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Kept deliberately small — 6 outer folds × inner CV already multiplies this
_RF_PARAM_GRID = {
    "rf__n_estimators":     [200, 400],
    "rf__max_depth":        [None, 35],
    "rf__min_samples_leaf": [1, 2],
    "rf__class_weight":     ["balanced"],
}


def _build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 6,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """
    Nested leave-one-driver-out CV. Inner GridSearchCV per outer fold uses
    only the remaining drivers — hyperparameters never tuned on held-out data.

    Returns the consensus-params pipeline (refitted on all data for SHAP),
    OOF hard predictions, and OOF class probabilities.
    """
    unique_drivers = sorted(np.unique(groups))
    n_outer        = min(n_splits, len(unique_drivers))
    n_classes      = len(np.unique(y))
    oof_preds      = np.full(len(y), -1, dtype=np.int64)
    oof_probs      = np.zeros((len(y), n_classes), dtype=np.float64)
    best_params_per_fold: list[dict] = []

    for fold_idx, held_out in enumerate(unique_drivers, 1):
        test_mask  = groups == held_out
        train_mask = ~test_mask
        X_tr, y_tr, g_tr = X[train_mask], y[train_mask], groups[train_mask]

        # Inner CV capped at 3 splits — enough to discriminate params without
        # multiplying cost by the full number of remaining drivers
        inner_cv = GroupKFold(n_splits=min(3, len(np.unique(g_tr))))
        search   = GridSearchCV(
            _build_rf_pipeline(), _RF_PARAM_GRID,
            cv=inner_cv, scoring="f1_macro", n_jobs=-1, verbose=0, refit=True,
        )
        search.fit(X_tr, y_tr, groups=g_tr)
        best_params_per_fold.append(search.best_params_)

        fold_probs            = search.best_estimator_.predict_proba(X[test_mask])
        oof_probs[test_mask]  = fold_probs
        oof_preds[test_mask]  = fold_probs.argmax(axis=1)
        print(f"[RF] Fold {fold_idx}/{n_outer} held out: {held_out} | "
              f"best: {search.best_params_}")

    assert (oof_preds >= 0).all()

    from collections import Counter
    consensus = {
        k: Counter(p[k] for p in best_params_per_fold).most_common(1)[0][0]
        for k in best_params_per_fold[0]
    }
    print(f"[RF] Consensus params: {consensus}")
    final = _build_rf_pipeline()
    final.set_params(**consensus)
    final.fit(X, y)
    return final, oof_preds, oof_probs


def save_rf(pipeline: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[RF] Saved to {path}")


def load_rf(path: str | Path) -> Pipeline:
    return joblib.load(path)
