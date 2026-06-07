"""
Inference module for IDX Stock Forecasting with TFT.
Handles test set prediction and submission file generation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
import torch
from torch import nn
from tqdm import tqdm  # type: ignore


def predict_test_batched(
    model: nn.Module,
    test_past: np.ndarray,
    test_future: np.ndarray,
    static_matrix: np.ndarray,
    lookback: int = 60,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-local-variables
    """
    Predict on test set using batched inference.

    For each timestep t in the test set (where t >= lookback), we predict
    for ALL 100 tickers simultaneously. Past data can reach back into
    the training set via the combined past array.

    Args:
        model: trained TFT model
        test_past: numpy array (n_combined, n_past) — concatenated train+test past features
        test_future: numpy array (n_combined, n_future) — concatenated train+test future features
        static_matrix: numpy array (100, n_static)
        lookback: int
        device: torch device
        batch_size: number of tickers to process per forward pass

    Returns:
        predictions: numpy array (n_test_timesteps, 100)
    """
    model.eval()
    n_tickers = static_matrix.shape[0]

    # Convert static to tensor once
    static_tensor = torch.tensor(static_matrix, dtype=torch.float32)  # (100, n_static)

    all_preds = []

    n_total = test_past.shape[0]

    with torch.no_grad():
        for t in tqdm(range(lookback, n_total), desc="  Predicting", leave=False):
            t_start = t - lookback

            # Past: shared across all tickers
            past_np = test_past[t_start:t]  # (lookback, n_past)
            past_t = torch.tensor(past_np, dtype=torch.float32).unsqueeze(
                0
            )  # (1, lookback, n_past)
            past_t = past_t.expand(n_tickers, -1, -1)  # (100, lookback, n_past)

            # Future: shared across all tickers
            future_np = test_future[t : t + 1]  # (1, n_future)
            future_t = torch.tensor(future_np, dtype=torch.float32).unsqueeze(
                0
            )  # (1, 1, n_future)
            future_t = future_t.expand(n_tickers, -1, -1)  # (100, 1, n_future)

            # Static: already per-ticker
            static_t = static_tensor  # (100, n_static)

            # Forward in batches if n_tickers is large
            tick_preds_list: list[np.ndarray] = []
            for start in range(0, n_tickers, batch_size):
                end = min(start + batch_size, n_tickers)

                pred = (
                    model(
                        static_t[start:end].to(device),
                        past_t[start:end].to(device),
                        future_t[start:end].to(device),
                    )
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )  # (batch,)

                tick_preds_list.append(pred)

            tick_preds = np.concatenate(tick_preds_list)  # (100,)
            all_preds.append(tick_preds)

    return np.array(all_preds)  # (n_test_timesteps, 100)


def create_submission(
    predictions: np.ndarray,
    target_tickers: Sequence[str],
    test_timestamps: Sequence[Any],
    sample_submission_path: str,
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """Create a submission CSV file."""
    # Build prediction lookup
    pred_lookup: dict[str, float] = {}
    for t_idx, ts in enumerate(test_timestamps):
        ts_str = str(ts).replace("T", " ").split(".", maxsplit=1)[0]
        for tick_idx, ticker in enumerate(target_tickers):
            pred_lookup[f"{ticker}_{ts_str}"] = float(predictions[t_idx, tick_idx])

    # Map predictions
    sub = pd.read_csv(sample_submission_path)
    sub["expected"] = sub["id"].map(lambda x: pred_lookup.get(str(x), 0.0))
    sub.to_csv(output_path, index=False)
    print(f"  Submission: {output_path} ({len(sub):,} rows)")
    return sub


def run_inference(
    model: nn.Module,
    data_dict: dict[str, Any],
    config: dict[str, Any],
    device: torch.device | str,
    output_path: str = "submission.csv",
) -> None:
    """End-to-end inference."""
    lookback = config["lookback"]
    combined_past = np.concatenate(
        [data_dict["train_past"], data_dict["test_past"]], axis=0
    )
    combined_future = np.concatenate(
        [data_dict["train_future"], data_dict["test_future"]], axis=0
    )

    # Inference view
    start_offset = max(0, data_dict["n_train"] - lookback)
    predictions = predict_test_batched(
        model=model,
        test_past=combined_past[start_offset:],
        test_future=combined_future[start_offset:],
        static_matrix=data_dict["static_matrix"],
        lookback=lookback,
        device=device,
        batch_size=100,
    )

    # Create submission
    create_submission(
        predictions=predictions,
        target_tickers=data_dict["target_tickers"],
        test_timestamps=data_dict["test_timestamps"],
        sample_submission_path=os.path.join(
            data_dict.get("data_dir", "dataset"), "sample_submission.csv"
        ),
        output_path=output_path,
    )
