from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN-BiLSTM attention model.")
    parser.add_argument("--model", "-m", type=str, default="SE", choices=["Base", "SE", "ECA", "CBAM", "HW"])
    parser.add_argument("--data", "-d", type=str, default="data/sample_tec.csv")
    parser.add_argument("--window", "-w", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="results")
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


def test() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data = StockDataset(args.data, args.window, is_test=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model)
    checkpoint_path = Path(args.checkpoint_dir) / f"{args.model}_best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    criterion = nn.MSELoss()
    eval_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, label in test_loader:
            out = model(data)
            loss = criterion(out, label)
            eval_loss += loss.item()
            y_true.extend(label.numpy()[:, -1])
            y_pred.extend(out.numpy()[:, -1])

    y_true_arr = test_data.inverse_transform_target(np.array(y_true).reshape(-1, 1))
    y_pred_arr = test_data.inverse_transform_target(np.array(y_pred).reshape(-1, 1))

    r2 = r2_score(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    mape = mean_absolute_percentage_error(y_true_arr, y_pred_arr) * 100

    limit = min(100, len(y_true_arr))
    plt.figure(figsize=(10, 5))
    plt.plot(y_true_arr[:limit], label="Real", linewidth=2.0)
    plt.plot(y_pred_arr[:limit], label="Predicted", linestyle="--", linewidth=2.0)
    plt.xlabel("Sample")
    plt.ylabel("TEC")
    plt.title(f"{args.model} Test Data Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.model}_prediction.png", dpi=300)
    plt.close()

    print(f"{args.model} eval loss: {eval_loss / max(1, len(test_loader)):.4f}")
    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")


if __name__ == "__main__":
    test()
