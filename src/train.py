from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader

from dataloader import StockDataset
from model import (
    CNNLSTMModel,
    CNNLSTMModel_CBAM,
    CNNLSTMModel_ECA,
    CNNLSTMModel_HW,
    CNNLSTMModel_SE,
)


matplotlib.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(0)
torch.manual_seed(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN-BiLSTM attention models for TEC prediction.")
    parser.add_argument("--model", "-m", type=str, default="SE", choices=["Base", "SE", "ECA", "CBAM", "HW"])
    parser.add_argument("--data", "-d", type=str, default="data/sample_tec.csv", help="Path to input CSV file.")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--window", "-w", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def build_model(name: str) -> nn.Module:
    model_dict = {
        "Base": CNNLSTMModel,
        "SE": CNNLSTMModel_SE,
        "ECA": CNNLSTMModel_ECA,
        "CBAM": CNNLSTMModel_CBAM,
        "HW": CNNLSTMModel_HW,
    }
    return model_dict[name]()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    dataset: StockDataset,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for data, label in loader:
            out = model(data)
            y_true.extend(label.numpy()[:, -1])
            y_pred.extend(out.numpy()[:, -1])

    y_true_arr = np.array(y_true).reshape(-1, 1)
    y_pred_arr = np.array(y_pred).reshape(-1, 1)
    y_true_arr = dataset.inverse_transform_target(y_true_arr)
    y_pred_arr = dataset.inverse_transform_target(y_pred_arr)
    return y_true_arr, y_pred_arr


def save_prediction_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "True_Value": y_true.flatten(),
            "Predicted_Value": y_pred.flatten(),
            "Absolute_Error": np.abs(y_true.flatten() - y_pred.flatten()),
        }
    )
    df.to_csv(path, index=False)


def plot_loss_curve(train_losses: list[float], test_losses: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_prediction(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = min(100, len(y_true))
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:limit], label="True Value", linewidth=2.0)
    plt.plot(y_pred[:limit], label="Predicted Value", linestyle="--", linewidth=2.0)
    plt.xlabel("Sample")
    plt.ylabel("TEC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def train() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_data = StockDataset(dataPath=str(data_path), window=args.window, is_test=False)
    test_data = StockDataset(dataPath=str(data_path), window=args.window, is_test=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Training set shape: {train_data.data.shape}")
    print(f"Test set shape: {test_data.data.shape}")

    model = build_model(args.model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    counter = 0
    train_losses: list[float] = []
    test_losses: list[float] = []
    checkpoint_path = checkpoint_dir / f"{args.model}_best.pth"
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for data, label in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, label in test_loader:
                out = model(data)
                loss = criterion(out, label)
                test_loss += loss.item()
        test_loss = test_loss / max(1, len(test_loader))
        test_losses.append(test_loss)

        print(f"Epoch {epoch + 1:03d} | train loss={train_loss:.6f} | val loss={test_loss:.6f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), checkpoint_path)
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    y_true_train, y_pred_train = collect_predictions(model, train_loader, train_data)
    y_true_test, y_pred_test = collect_predictions(model, test_loader, test_data)

    save_prediction_csv(output_dir / "train_prediction_results.csv", y_true_train, y_pred_train)
    save_prediction_csv(output_dir / "test_prediction_results.csv", y_true_test, y_pred_test)
    plot_loss_curve(train_losses, test_losses, output_dir / "loss_curve.png")
    plot_prediction(y_true_test, y_pred_test, output_dir / "prediction_vs_true_test.png", f"{args.model} Test Set")

    metrics = evaluate_predictions(y_true_test, y_pred_test)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(pd.Series(metrics).to_json(indent=2), encoding="utf-8")

    print(f"Training took {(time.time() - start_time) / 60:.2f} minutes")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    train()
