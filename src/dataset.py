"""
dataset.py — PyTorch Dataset and DataLoader utilities for IDX Stock Forecasting TFT.

Provides:
    - IDXDataset: A map-style Dataset where each sample represents ONE ticker
      at ONE timestep.  Past/future market data are shared across tickers;
      only static features and regression targets differ per ticker.
    - create_dataloaders: Convenience function that performs a temporal
      train/val split, constructs IDXDataset instances, and wraps them in
      DataLoaders ready for training, validation, and test inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Dataset
class IDXDataset(Dataset):
    # pylint: disable=too-many-instance-attributes
    """PyTorch Dataset for the IDX stock-forecasting task.

    Each sample is identified by a ``(timestep, ticker)`` pair.  The total
    number of valid samples equals ``(n_timesteps - lookback) × n_tickers``.

    Parameters
    ----------
    past_data : np.ndarray, shape (n_timesteps, n_past_features)
        Pre-processed historical / known-past features shared across all
        tickers.
    future_data : np.ndarray, shape (n_timesteps, n_future_features)
        Pre-processed known-future features shared across all tickers.
    static_matrix : np.ndarray, shape (n_tickers, n_static)
        Static (time-invariant) features per ticker.  Typically
        ``n_tickers = 100`` and ``n_static = 3``.
    targets : np.ndarray or None, shape (n_timesteps, n_tickers)
        Regression targets.  Set to ``None`` for test / inference mode.
    lookback : int
        Number of historical timesteps fed to the model (default ``60``).
    mode : str
        ``'train'`` or ``'test'``.  When ``'test'``, targets are not
        included in the returned dictionary.
    """

    def __init__(
        self,
        past_data: np.ndarray,
        future_data: np.ndarray,
        static_matrix: np.ndarray,
        targets: np.ndarray | None = None,
        lookback: int = 60,
        mode: str = "train",
    ) -> None:
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__()

        self.past_data = past_data.astype(np.float32)
        self.future_data = future_data.astype(np.float32)
        self.static_matrix = static_matrix.astype(np.float32)
        self.targets = targets.astype(np.float32) if targets is not None else None
        self.lookback = lookback
        self.mode = mode

        self.n_timesteps: int = past_data.shape[0]
        self.n_tickers: int = static_matrix.shape[0]

        # Valid starting points are t = lookback, lookback+1, …, n_timesteps-1
        self.valid_t = list(range(lookback, self.n_timesteps))
        self.n_valid_t: int = len(self.valid_t)

    # Dataset interface
    def __len__(self) -> int:
        """Total samples = (n_timesteps - lookback) * n_tickers."""
        return self.n_valid_t * self.n_tickers

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample for the given linear index (timestep, ticker).

        Returns
        -------
        dict with keys:
            ``'static'``  — shape ``(n_static,)``
            ``'past'``    — shape ``(lookback, n_past_features)``
            ``'future'``  — shape ``(1, n_future_features)``
            ``'target'``  — shape ``()`` (scalar, train mode only)
        """
        t_idx = idx // self.n_tickers
        ticker_idx = idx % self.n_tickers
        t = self.valid_t[t_idx]

        sample: dict[str, torch.Tensor] = {
            "static": torch.from_numpy(self.static_matrix[ticker_idx]),  # (n_static,)
            "past": torch.from_numpy(
                self.past_data[t - self.lookback : t]
            ),  # (lookback, n_past)
            "future": torch.from_numpy(self.future_data[t : t + 1]),  # (1, n_future)
        }

        if self.mode == "train" and self.targets is not None:
            sample["target"] = torch.tensor(
                self.targets[t, ticker_idx], dtype=torch.float32
            )

        return sample


# DataLoader factory
def create_dataloaders(
    data_dict: dict[str, Any],
    lookback: int = 60,
    batch_size: int = 256,
    val_split: float = 0.8,
    num_workers: int = 0,
    is_distributed: bool = False,
) -> dict[str, Any]:
    # pylint: disable=too-many-local-variables
    """Build train / val / test DataLoaders from the preprocessed data dict.

    The training data is split **temporally** (no random shuffling for the
    split itself).  The first ``val_split`` fraction of timesteps becomes
    the training set; the remainder is used for validation.

    Parameters
    ----------
    data_dict : dict
        Dictionary returned by ``preprocess_data()`` from
        ``preprocessing.py``.  Expected keys:

        - ``train_past``    — ``(n_train_timesteps, n_past_features)``
        - ``train_future``  — ``(n_train_timesteps, n_future_features)``
        - ``train_targets`` — ``(n_train_timesteps, n_tickers)``
        - ``test_past``     — ``(n_test_timesteps, n_past_features)``
        - ``test_future``   — ``(n_test_timesteps, n_future_features)``
        - ``static_matrix`` — ``(n_tickers, n_static)``

    lookback : int
        Number of historical timesteps per sample (default ``60``).
    batch_size : int
        Mini-batch size (default ``256``).
    val_split : float
        Fraction of training timesteps used for actual training
        (default ``0.8``).  The rest is validation.
    num_workers : int
        Number of DataLoader worker processes (default ``0``).

    Returns
    -------
    dict
        ``train_loader``, ``val_loader``, ``test_loader`` — DataLoaders.
        ``split_idx`` — the integer index at which the temporal split occurs.
    """

    # ----- scale batch size if distributed ----------------------------------
    if is_distributed:
        import torch.distributed as dist

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        batch_size = max(1, batch_size // world_size)

    # ----- unpack ----------------------------------------------------------
    train_past: np.ndarray = data_dict["train_past"]
    train_future: np.ndarray = data_dict["train_future"]
    train_targets: np.ndarray = data_dict["train_targets"]
    test_past: np.ndarray = data_dict["test_past"]
    test_future: np.ndarray = data_dict["test_future"]
    static_matrix: np.ndarray = data_dict["static_matrix"]

    # ----- temporal split --------------------------------------------------
    n_train_timesteps = train_past.shape[0]
    split_idx = int(n_train_timesteps * val_split)

    train_past_split = train_past[:split_idx]
    train_future_split = train_future[:split_idx]
    train_targets_split = train_targets[:split_idx]

    val_past_split = train_past[split_idx:]
    val_future_split = train_future[split_idx:]
    val_targets_split = train_targets[split_idx:]

    # ----- datasets --------------------------------------------------------
    train_dataset = IDXDataset(
        past_data=train_past_split,
        future_data=train_future_split,
        static_matrix=static_matrix,
        targets=train_targets_split,
        lookback=lookback,
        mode="train",
    )

    val_dataset = IDXDataset(
        past_data=val_past_split,
        future_data=val_future_split,
        static_matrix=static_matrix,
        targets=val_targets_split,
        lookback=lookback,
        mode="train",
    )

    test_dataset = IDXDataset(
        past_data=test_past,
        future_data=test_future,
        static_matrix=static_matrix,
        targets=None,
        lookback=lookback,
        mode="test",
    )

    # ----- datasets --------------------------------------------------------
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            f"  Dataset: {len(train_dataset):,} train timesteps | {len(val_dataset):,} val timesteps"
        )

    # ----- dataloaders -----------------------------------------------------
    pin_memory = torch.cuda.is_available()

    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "split_idx": split_idx,
    }
