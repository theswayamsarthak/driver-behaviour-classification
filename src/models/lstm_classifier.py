from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ── architectures ──────────────────────────────────────────────────────────────

class DriverLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.arch = "lstm"
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out        = hidden_size * (2 if bidirectional else 1)
        self.norm       = nn.LayerNorm(lstm_out)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.classifier(self.dropout(self.norm(out.mean(dim=1))))


class DriverCNNLSTM(nn.Module):
    """Conv1d feature extractor followed by BiLSTM temporal encoder."""
    def __init__(
        self,
        input_size: int,
        conv_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.arch = "cnn_lstm"
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels), nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out        = hidden_size * (2 if bidirectional else 1)
        self.norm       = nn.LayerNorm(lstm_out)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(c)
        return self.classifier(self.dropout(self.norm(out.mean(dim=1))))


def build_model(
    arch: str, input_size: int, hidden_size: int, num_layers: int,
    num_classes: int, dropout: float, conv_channels: int = 64,
) -> DriverLSTM | DriverCNNLSTM:
    if arch == "cnn_lstm":
        return DriverCNNLSTM(input_size, conv_channels, hidden_size,
                             num_layers, num_classes, dropout)
    return DriverLSTM(input_size, hidden_size, num_layers, num_classes, dropout)


# ── loss ───────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


# ── internal helpers ───────────────────────────────────────────────────────────

def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size, shuffle=shuffle,
    )


def _class_weights(y: np.ndarray, n: int, device: torch.device) -> torch.Tensor:
    counts  = np.bincount(y, minlength=n).astype(float)
    weights = counts.sum() / (n * counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_one_model(
    X_train: np.ndarray, y_train: np.ndarray,
    arch: str, num_classes: int, hidden_size: int, num_layers: int,
    conv_channels: int, dropout: float, epochs: int, lr: float,
    batch_size: int, use_focal_loss: bool, focal_gamma: float,
    device: torch.device, log_prefix: str = "",
) -> tuple[DriverLSTM | DriverCNNLSTM, np.ndarray]:
    n, t, c   = X_train.shape
    scaler    = StandardScaler()
    X_s       = scaler.fit_transform(X_train.reshape(-1, c)).reshape(n, t, c)
    loader    = _make_loader(X_s, y_train, batch_size, shuffle=True)
    model     = build_model(arch, c, hidden_size, num_layers, num_classes,
                            dropout, conv_channels).to(device)
    cw        = _class_weights(y_train, num_classes, device)
    criterion = FocalLoss(gamma=focal_gamma, weight=None) if use_focal_loss \
                else nn.CrossEntropyLoss(weight=cw)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    losses    = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            running += loss.item() * len(yb)
        scheduler.step()
        losses.append(running / len(loader.dataset))
        if epoch == epochs or epoch % 10 == 0:
            print(f"  {log_prefix}epoch {epoch:3d}/{epochs} | loss={losses[-1]:.4f}")

    # Convergence check: if loss at final epoch is more than 5% below loss at
    # the halfway point, the model is still learning — suggest more epochs.
    if len(losses) >= 4:
        halfway = losses[len(losses) // 2]
        final   = losses[-1]
        if halfway > 0 and (halfway - final) / halfway > 0.05:
            print(f"  [LSTM] WARNING: loss still declining ({halfway:.4f} → {final:.4f}, "
                  f"{100*(halfway-final)/halfway:.1f}% drop in second half). "
                  f"Consider increasing --lstm_epochs beyond {epochs}.")

    return model, scaler, losses


# ── public training API ────────────────────────────────────────────────────────

def cross_validate_lstm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    num_classes: int = 3,
    arch: str = "lstm",
    hidden_size: int = 128,
    num_layers: int = 2,
    conv_channels: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    dropout: float = 0.3,
    use_focal_loss: bool = False,
    focal_gamma: float = 1.0,
    seed: int = 42,
    device: Optional[str] = None,
    return_fold_models: bool = False,
) -> tuple:
    """
    Leave-one-driver-out cross-validation. Trains one model per held-out
    driver, assembles genuine out-of-fold predictions and probabilities.

    When return_fold_models=True, also returns the list of (model, scaler,
    test_mask) tuples so SHAP can reuse the same models without retraining.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    _dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    unique_drivers = sorted(np.unique(groups))
    oof_preds  = np.full(len(y), -1, dtype=np.int64)
    oof_probs  = np.zeros((len(y), num_classes), dtype=np.float64)
    histories  = []
    fold_models: list[tuple] = []

    for fold_idx, held_out in enumerate(unique_drivers, 1):
        print(f"\n[LSTM-CV] Fold {fold_idx}/{len(unique_drivers)} — held out: {held_out}")
        test_mask, train_mask = groups == held_out, groups != held_out

        model, scaler, losses = _train_one_model(
            X[train_mask], y[train_mask], arch, num_classes,
            hidden_size, num_layers, conv_channels, dropout,
            epochs, lr, batch_size, use_focal_loss, focal_gamma, _dev,
        )

        n, t, c  = X[test_mask].shape
        X_test_s = scaler.transform(X[test_mask].reshape(-1, c)).reshape(n, t, c)
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test_s, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False,
        )

        model.eval()
        fold_preds, fold_probs_list = [], []
        with torch.no_grad():
            for (xb,) in test_loader:
                probs = torch.softmax(model(xb.to(_dev)), dim=1)
                fold_preds.extend(probs.argmax(1).cpu().numpy().tolist())
                fold_probs_list.append(probs.cpu().numpy())

        oof_preds[test_mask] = np.array(fold_preds, dtype=np.int64)
        oof_probs[test_mask] = np.concatenate(fold_probs_list, axis=0)
        histories.append({"driver": held_out, "train_loss": losses})

        if return_fold_models:
            fold_models.append((model, scaler, test_mask))

    assert (oof_preds >= 0).all()

    if return_fold_models:
        return oof_preds, oof_probs, histories, fold_models
    return oof_preds, oof_probs, histories


def train_lstm_final(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    num_classes: int = 3,
    arch: str = "lstm",
    hidden_size: int = 128,
    num_layers: int = 2,
    conv_channels: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    dropout: float = 0.3,
    use_focal_loss: bool = False,
    focal_gamma: float = 1.0,
    seed: int = 42,
    device: Optional[str] = None,
) -> tuple[DriverLSTM | DriverCNNLSTM, StandardScaler, dict]:
    """Train on all drivers. Used for SHAP explainability and deployment only."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    _dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[LSTM-Final] Training deployment model on all {len(np.unique(groups))} drivers...")

    model, scaler, losses = _train_one_model(
        X, y, arch, num_classes, hidden_size, num_layers,
        conv_channels, dropout, epochs, lr, batch_size,
        use_focal_loss, focal_gamma, _dev,
    )
    return model, scaler, {"train_loss": losses}


# ── inference ──────────────────────────────────────────────────────────────────

def predict_lstm(
    model: DriverLSTM | DriverCNNLSTM,
    X: np.ndarray,
    scaler: StandardScaler,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    _dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(_dev)
    n, t, c = X.shape
    Xs = scaler.transform(X.reshape(-1, c)).reshape(n, t, c)
    loader = DataLoader(
        TensorDataset(torch.tensor(Xs, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )
    preds: list[int] = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.extend(model(xb.to(_dev)).argmax(1).cpu().numpy().tolist())
    return np.array(preds, dtype=np.int64)


# ── persistence ────────────────────────────────────────────────────────────────

def _get_config(model: DriverLSTM | DriverCNNLSTM) -> dict:
    cfg: dict = {"arch": model.arch, "dropout": model.dropout.p,
                 "num_classes": model.classifier.out_features}
    if model.arch == "cnn_lstm":
        cfg.update({
            "input_size":    model.conv[0].in_channels,
            "conv_channels": model.conv[0].out_channels,
            "hidden_size":   model.lstm.hidden_size,
            "num_layers":    model.lstm.num_layers,
            "bidirectional": model.lstm.bidirectional,
        })
    else:
        cfg.update({
            "input_size":    model.lstm.input_size,
            "hidden_size":   model.lstm.hidden_size,
            "num_layers":    model.lstm.num_layers,
            "bidirectional": model.lstm.bidirectional,
        })
    return cfg


def save_lstm(
    model: DriverLSTM | DriverCNNLSTM, scaler: StandardScaler, path: str | Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": _get_config(model)}, path)
    joblib.dump(scaler, path.with_suffix(".scaler.pkl"))
    print(f"[LSTM] Saved to {path} + {path.with_suffix('.scaler.pkl')}")


def load_lstm(path: str | Path) -> tuple[DriverLSTM | DriverCNNLSTM, StandardScaler]:
    path       = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg        = checkpoint["config"]
    model      = build_model(
        arch=cfg["arch"], input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"], dropout=cfg["dropout"],
        conv_channels=cfg.get("conv_channels", 64),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model, joblib.load(path.with_suffix(".scaler.pkl"))
