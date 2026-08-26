from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

LABEL_MAP: dict[str, int] = {"AGGRESSIVE": 0, "NORMAL": 1, "DROWSY": 2}
LABEL_NAMES: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

SIGNAL_COLS: list[str] = [
    "speed", "acc_lon", "acc_lat", "acc_vert", "gyro_x", "gyro_y", "gyro_z",
]

SAMPLE_RATE_HZ: int = 10
_LABEL_RE  = re.compile(r"-(AGGRESSIVE|NORMAL[12]?|DROWSY)-", re.IGNORECASE)
_DRIVER_RE = re.compile(r"^D\d+$", re.IGNORECASE)


def _parse_folder_label(folder_name: str) -> str | None:
    m = _LABEL_RE.search(folder_name)
    if m is None:
        return None
    raw = m.group(1).upper()
    return "NORMAL" if raw.startswith("NORMAL") else raw


def _load_semantic_labels(
    path: Path, session_label_int: int
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        df = pd.read_csv(
            path, sep=r"\s+", header=None, usecols=[0, 1],
            names=["timestamp", "sem_label"], on_bad_lines="skip", dtype=float,
        )
        df = df.dropna().reset_index(drop=True)
        if len(df) < 2:
            return None
    except Exception:
        return None

    raw = df["sem_label"].values.astype(int)
    unique_vals, counts = np.unique(raw, return_counts=True)
    mode_raw = int(unique_vals[np.argmax(counts)])
    mapping: dict[int, int] = {mode_raw: session_label_int}
    remaining_raw  = sorted(v for v in unique_vals if v != mode_raw)
    remaining_ours = sorted(l for l in [0, 1, 2] if l != session_label_int)
    for rv, ol in zip(remaining_raw, remaining_ours):
        mapping[rv] = ol
    mapped = np.array([mapping.get(int(v), session_label_int) for v in raw])
    return df["timestamp"].values, mapped


def _load_accelerometers(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(
            path, sep=r"\s+", header=None, usecols=[0, 1, 2, 3, 4, 5, 6],
            names=["timestamp", "acc_lon", "acc_lat", "acc_vert", "gyro_x", "gyro_y", "gyro_z"],
            on_bad_lines="skip", dtype=float,
        )
    except Exception:
        return None
    df = df.dropna().reset_index(drop=True)
    return df if len(df) >= 2 else None


def _load_gps_speed(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        df = pd.read_csv(
            path, sep=r"\s+", header=None, usecols=[0, 4],
            names=["timestamp", "speed"], on_bad_lines="skip", dtype=float,
        )
    except Exception:
        return None
    df = df.dropna().reset_index(drop=True)
    return (df["timestamp"].values, df["speed"].values) if len(df) >= 2 else None


def _merge_signals(
    accel_df: pd.DataFrame, gps_ts: np.ndarray, gps_speed: np.ndarray
) -> pd.DataFrame:
    t = accel_df["timestamp"].values
    speed = np.interp(t, gps_ts, gps_speed).astype(np.float32)
    out = accel_df.copy()
    out.insert(0, "speed", speed)
    return out[SIGNAL_COLS].reset_index(drop=True)


def _sliding_windows(
    signal_df: pd.DataFrame,
    label_arr: np.ndarray,
    window_samples: int,
    step_samples: int,
) -> list[tuple[pd.DataFrame, int]]:
    results = []
    n, start = len(signal_df), 0
    while start + window_samples <= n:
        win_signals = signal_df.iloc[start : start + window_samples].copy()
        win_labels  = label_arr[start : start + window_samples]
        majority    = int(np.argmax(np.bincount(win_labels, minlength=3)))
        results.append((win_signals.reset_index(drop=True), majority))
        start += step_samples
    return results


def _iter_sessions(data_dir: Path) -> Generator[tuple[Path, str, str], None, None]:
    for driver_dir in sorted(data_dir.iterdir()):
        if not driver_dir.is_dir() or not _DRIVER_RE.match(driver_dir.name):
            continue
        for session_dir in sorted(driver_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            label = _parse_folder_label(session_dir.name)
            if label is None or label not in LABEL_MAP:
                continue
            yield session_dir, label, driver_dir.name


def load_dataset(
    data_dir: str | Path,
    window_sec: float = 5.0,
    overlap: float = 0.5,
    use_semantic_labels: bool = True,
) -> tuple[list[pd.DataFrame], list[int], list[str]]:
    """
    Parse UAH-DriveSet and return sliding windows with labels.

    Returns windows (list of DataFrames), integer labels, and driver IDs
    (used as groups for leave-one-driver-out cross-validation).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    window_samples = int(window_sec * SAMPLE_RATE_HZ)
    step_samples   = max(1, int(window_samples * (1.0 - overlap)))

    windows: list[pd.DataFrame] = []
    labels:  list[int]          = []
    drivers: list[str]          = []
    skipped, valid_sessions     = 0, 0
    sem_used, sem_missing, sem_failed_parse = 0, 0, 0

    for session_dir, label_str, driver_id in _iter_sessions(data_dir):
        accel_file = session_dir / "RAW_ACCELEROMETERS.txt"
        if not accel_file.exists():
            skipped += 1
            continue

        accel_df = _load_accelerometers(accel_file)
        if accel_df is None or len(accel_df) < window_samples:
            skipped += 1
            continue

        valid_sessions += 1
        gps_file = session_dir / "RAW_GPS.txt"
        gps = _load_gps_speed(gps_file) if gps_file.exists() else None

        if gps is not None:
            signal_df = _merge_signals(accel_df, gps[0], gps[1])
        else:
            accel_df.insert(0, "speed", np.zeros(len(accel_df), dtype=np.float32))
            signal_df = accel_df[SIGNAL_COLS].reset_index(drop=True)

        session_label_int = LABEL_MAP[label_str]
        sem_file = session_dir / "SEMANTIC_FINAL.txt"
        per_sample_labels: np.ndarray | None = None

        if use_semantic_labels:
            if not sem_file.exists():
                sem_missing += 1
            else:
                sem = _load_semantic_labels(sem_file, session_label_int)
                if sem is not None:
                    sem_ts, sem_labs = sem
                    indices = np.searchsorted(sem_ts, accel_df["timestamp"].values, side="right") - 1
                    indices = np.clip(indices, 0, len(sem_labs) - 1)
                    per_sample_labels = sem_labs[indices]
                    sem_used += 1
                else:
                    sem_failed_parse += 1

        if per_sample_labels is None:
            per_sample_labels = np.full(len(signal_df), session_label_int, dtype=np.int64)

        for win_df, majority_label in _sliding_windows(
            signal_df, per_sample_labels, window_samples, step_samples
        ):
            windows.append(win_df)
            labels.append(majority_label)
            drivers.append(driver_id)

    if not windows:
        raise RuntimeError(
            f"No valid sessions found under {data_dir}.\n"
            f"  Skipped {skipped} sessions.\n"
            f"  Run diagnose.py to verify your dataset layout."
        )

    if skipped:
        print(f"[Data] Skipped {skipped} sessions (missing RAW_ACCELEROMETERS.txt or too short)")

    if use_semantic_labels:
        print(f"[Data] SEMANTIC_FINAL.txt used in {sem_used}/{valid_sessions} sessions "
              f"(missing: {sem_missing}, unparseable: {sem_failed_parse})")
        if sem_used == 0 and valid_sessions > 0:
            print("[Data] WARNING: no per-second labels found — falling back to session-level "
                  "folder labels for all windows. Pass --no_semantic to suppress this warning.")

    return windows, labels, drivers


def windows_to_array(windows: list[pd.DataFrame]) -> np.ndarray:
    """Stack list of (T, C) DataFrames into (N, T, C) float32 array."""
    return np.stack([w.values.astype(np.float32) for w in windows], axis=0)
