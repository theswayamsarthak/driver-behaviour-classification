from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    "speed_mean", "speed_std", "speed_max", "speed_range",
    "acc_lon_mean", "acc_lon_std", "acc_lon_max_abs", "acc_lon_rms", "acc_lon_jerk_std",
    "acc_lat_mean", "acc_lat_std", "acc_lat_max_abs", "acc_lat_rms", "acc_lat_jerk_std",
    "acc_vert_std", "acc_vert_max_abs",
    "gyro_z_mean_abs", "gyro_z_std", "gyro_z_max_abs", "gyro_z_reversal_count",
    "gyro_x_std",
    "hard_brake_events", "hard_accel_events", "hard_cornering_events",
    "max_decel", "speed_acc_lon_corr", "lateral_over_longitudinal_rms",
]

assert len(FEATURE_NAMES) == 27

_HARD_BRAKE_G    = -0.3
_HARD_ACCEL_G    =  0.25
_HARD_CORNERING  =  10.0
_GYRO_Z_REVERSAL =  5.0
_DT_SEC          =  0.1


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def _jerk_std(arr: np.ndarray) -> float:
    jerk = np.diff(arr) / _DT_SEC
    return float(np.std(jerk)) if len(jerk) > 0 else 0.0


def _reversal_count(arr: np.ndarray, threshold: float) -> int:
    sign: bool | None = None
    count = 0
    for v in arr:
        if abs(v) < threshold:
            continue
        cur = v > 0
        if sign is not None and cur != sign:
            count += 1
        sign = cur
    return count


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_features(window: pd.DataFrame) -> np.ndarray:
    speed    = window["speed"].values.astype(float)
    acc_lon  = window["acc_lon"].values.astype(float)
    acc_lat  = window["acc_lat"].values.astype(float)
    acc_vert = window["acc_vert"].values.astype(float)
    gyro_x   = window["gyro_x"].values.astype(float)
    gyro_z   = window["gyro_z"].values.astype(float)

    acc_lon_rms = _rms(acc_lon)
    acc_lat_rms = _rms(acc_lat)

    return np.array([
        np.mean(speed), np.std(speed), float(np.max(speed)),
        float(np.max(speed) - np.min(speed)),
        np.mean(acc_lon), np.std(acc_lon), float(np.max(np.abs(acc_lon))),
        acc_lon_rms, _jerk_std(acc_lon),
        np.mean(acc_lat), np.std(acc_lat), float(np.max(np.abs(acc_lat))),
        acc_lat_rms, _jerk_std(acc_lat),
        np.std(acc_vert), float(np.max(np.abs(acc_vert))),
        float(np.mean(np.abs(gyro_z))), np.std(gyro_z), float(np.max(np.abs(gyro_z))),
        float(_reversal_count(gyro_z, threshold=_GYRO_Z_REVERSAL)),
        np.std(gyro_x),
        float(np.sum(acc_lon < _HARD_BRAKE_G)),
        float(np.sum(acc_lon > _HARD_ACCEL_G)),
        float(np.sum(np.abs(gyro_z) > _HARD_CORNERING)),
        float(np.min(acc_lon)),
        _safe_corr(speed, acc_lon),
        acc_lat_rms / (acc_lon_rms + 1e-8),
    ], dtype=np.float32)


def build_feature_matrix(windows: list[pd.DataFrame]) -> np.ndarray:
    return np.stack([extract_features(w) for w in windows], axis=0)
