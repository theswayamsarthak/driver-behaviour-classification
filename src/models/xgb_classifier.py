from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_XGB_PARAM_GRID = {
    "xgb__n_estimators":     [200, 400],
    "xgb__max_depth":        [4, 6],
    "xgb__learning_rate":    [0.05, 0.1],
    "xgb__subsample":        [0.8],
    "xgb__colsample_bytree": [0.8],
}


def _build_xgb_pipeline(num_classes: int) -> Pipeline:
    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            objective=objective,
            num_class=num_classes if num_classes > 2 else None,
            eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 6,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """Nested LODO CV for XGBoost — same structure as train_random_forest."""
    num_classes    = len(np.unique(y))
    unique_drivers = sorted(np.unique(groups))
    n_outer        = min(n_splits, len(unique_drivers))
    oof_preds      = np.full(len(y), -1, dtype=np.int64)
    oof_probs      = np.zeros((len(y), num_classes), dtype=np.float64)
    best_params_per_fold: list[dict] = []

    for fold_idx, held_out in enumerate(unique_drivers, 1):
        test_mask  = groups == held_out
        train_mask = ~test_mask
        X_tr, y_tr, g_tr = X[train_mask], y[train_mask], groups[train_mask]

        inner_cv = GroupKFold(n_splits=min(3, len(np.unique(g_tr))))
        search   = GridSearchCV(
            _build_xgb_pipeline(num_classes), _XGB_PARAM_GRID,
            cv=inner_cv, scoring="f1_macro", n_jobs=-1, verbose=0, refit=True,
        )
        search.fit(X_tr, y_tr, groups=g_tr)
        best_params_per_fold.append(search.best_params_)

        fold_probs            = search.best_estimator_.predict_proba(X[test_mask])
        oof_probs[test_mask]  = fold_probs
        oof_preds[test_mask]  = fold_probs.argmax(axis=1)
        print(f"[XGB] Fold {fold_idx}/{n_outer} held out: {held_out} | "
              f"best: {search.best_params_}")

    assert (oof_preds >= 0).all()

    from collections import Counter
    consensus = {
        k: Counter(p[k] for p in best_params_per_fold).most_common(1)[0][0]
        for k in best_params_per_fold[0]
    }
    print(f"[XGB] Consensus params: {consensus}")
    final = _build_xgb_pipeline(num_classes)
    final.set_params(**consensus)
    final.fit(X, y)
    return final, oof_preds, oof_probs


def save_xgb(pipeline: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[XGB] Saved to {path}")


def load_xgb(path: str | Path) -> Pipeline:
    return joblib.load(path)
