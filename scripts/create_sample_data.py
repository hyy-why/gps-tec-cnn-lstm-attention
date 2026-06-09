from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    n = 240
    t = np.arange(n)

    data = {}
    for i in range(12):
        seasonal = np.sin(t / (8 + i)) + 0.3 * np.cos(t / (13 + i))
        noise = rng.normal(0, 0.1, size=n)
        data[f"feature_{i + 1}"] = seasonal + noise + i * 0.05

    feature_matrix = np.column_stack([data[f"feature_{i + 1}"] for i in range(12)])
    weights = np.linspace(0.2, 1.0, 12)
    tec = feature_matrix @ weights / weights.sum()
    tec += 0.2 * np.sin(t / 15) + rng.normal(0, 0.05, size=n)
    data["TEC"] = tec

    out = Path("data/sample_tec.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
