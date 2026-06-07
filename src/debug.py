"""
debug.py — Logical and numerical checking modules for IDX Stock Forecasting TFT.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def run_overfit_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    epochs: int = 100,
    lr: float = 0.005,
) -> None:
    # pylint: disable=too-many-locals
    """Train the model on a single batch to check if loss converges to near 0."""
    print("\n>>> STARTING LOGICAL TEST: OVERFIT SINGLE BATCH")
    model.train()

    # Get first batch
    batch = next(iter(loader))

    # Setup optimizer (high LR, no weight decay/regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Unpack batch and reshape
    B_sz, n_tickers, n_static = batch["static"].shape
    lookback = batch["past"].shape[1]
    n_past = batch["past"].shape[2]
    n_future = batch["future"].shape[2]

    static_x = batch["static"].view(B_sz * n_tickers, n_static).to(device)

    past_x = batch["past"].unsqueeze(1).expand(-1, n_tickers, -1, -1)
    past_x = past_x.reshape(B_sz * n_tickers, lookback, n_past).to(device)

    future_x = batch["future"].unsqueeze(1).expand(-1, n_tickers, -1, -1)
    future_x = future_x.reshape(B_sz * n_tickers, 1, n_future).to(device)

    targets = batch["target"].view(B_sz * n_tickers).to(device)

    initial_loss = None
    final_loss = None

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        predictions = model(static_x, past_x, future_x).squeeze(-1)
        loss = criterion(predictions, targets)
        loss.backward()

        # Clip grad norm to verify clip grad doesn't break
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.get("grad_clip", 1.0)
        )
        optimizer.step()

        if epoch == 1:
            initial_loss = loss.item()
        final_loss = loss.item()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:03d}/{epochs:03d} | loss: {final_loss:.8f}")

    print(f"  Initial Loss: {initial_loss:.6f} -> Final Loss: {final_loss:.8f}")
    if final_loss < 1e-4:
        print("  [PASS] Overfitting test succeeded!")
    else:
        print("  [FAIL] Overfitting test failed. Loss did not converge to near 0.")


def run_causality_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    # pylint: disable=too-many-locals
    """Verify that predictions are responsive to past features and future features, and observe behavior."""
    print("\n>>> STARTING LOGICAL TEST: TIME CAUSALITY & RESPONSIVENESS")
    model.eval()

    batch = next(iter(loader))
    B_sz, n_tickers, n_static = batch["static"].shape
    lookback = batch["past"].shape[1]
    n_past = batch["past"].shape[2]
    n_future = batch["future"].shape[2]

    static_x = batch["static"].view(B_sz * n_tickers, n_static).to(device)
    past_x = batch["past"].unsqueeze(1).expand(-1, n_tickers, -1, -1)
    past_x = past_x.reshape(B_sz * n_tickers, lookback, n_past).to(device)
    future_x = batch["future"].unsqueeze(1).expand(-1, n_tickers, -1, -1)
    future_x = future_x.reshape(B_sz * n_tickers, 1, n_future).to(device)

    with torch.no_grad():
        preds_orig = model(static_x, past_x, future_x).cpu()

    # Modify last timestep of past data (t-1)
    past_mod_last = past_x.clone()
    past_mod_last[:, -1, :] += 1.0
    with torch.no_grad():
        preds_mod_last = model(static_x, past_mod_last, future_x).cpu()

    # Modify first timestep of past data (t-lookback)
    past_mod_first = past_x.clone()
    past_mod_first[:, 0, :] += 1.0
    with torch.no_grad():
        preds_mod_first = model(static_x, past_mod_first, future_x).cpu()

    # Modify future data
    future_mod = future_x.clone()
    future_mod[:, :, :] += 1.0
    with torch.no_grad():
        preds_mod_future = model(static_x, past_x, future_mod).cpu()

    diff_last = torch.abs(preds_orig - preds_mod_last).mean().item()
    diff_first = torch.abs(preds_orig - preds_mod_first).mean().item()
    diff_future = torch.abs(preds_orig - preds_mod_future).mean().item()

    print(
        f"  Mean absolute change in predictions when modifying t-1 (past): {diff_last:.8f}"
    )
    print(
        f"  Mean absolute change in predictions when modifying t-lookback (past): {diff_first:.8f}"
    )
    print(
        f"  Mean absolute change in predictions when modifying future features: {diff_future:.8f}"
    )

    # Check if the model has zero sensitivity to inputs (dead networks / underflow)
    if diff_last > 1e-6 and diff_first > 1e-6 and diff_future > 1e-6:
        print("  [PASS] Causality and responsiveness test succeeded!")
    else:
        print(
            "  [FAIL] Causality test failed. Predictions did not change when input features were modified."
        )


def run_permutation_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    # pylint: disable=too-many-locals
    """Verify that permuting the order of tickers in the input permutes predictions identically (no cross-ticker leakage)."""
    print("\n>>> STARTING LOGICAL TEST: TICKER PERMUTATION INVARIANCE")
    model.eval()

    batch = next(iter(loader))
    B_sz, n_tickers, n_static = batch["static"].shape
    lookback = batch["past"].shape[1]
    n_past = batch["past"].shape[2]
    n_future = batch["future"].shape[2]

    static_x = batch["static"].view(B_sz * n_tickers, n_static).to(device)
    past_x = batch["past"].unsqueeze(1).expand(-1, n_tickers, -1, -1)
    past_x = past_x.reshape(B_sz * n_tickers, lookback, n_past).to(device)
    future_x = batch["future"].unsqueeze(1).expand(-1, n_tickers, -1, -1)
    future_x = future_x.reshape(B_sz * n_tickers, 1, n_future).to(device)

    with torch.no_grad():
        preds_orig = model(static_x, past_x, future_x).view(B_sz, n_tickers).cpu()

    # Permute tickers (reverse ordering)
    perm = torch.arange(n_tickers - 1, -1, -1)

    static_perm = batch["static"][:, perm, :]
    static_perm_x = static_perm.view(B_sz * n_tickers, n_static).to(device)

    past_perm = batch["past"].unsqueeze(1).expand(-1, n_tickers, -1, -1)[:, perm, :, :]
    past_perm_x = past_perm.reshape(B_sz * n_tickers, lookback, n_past).to(device)

    future_perm = (
        batch["future"].unsqueeze(1).expand(-1, n_tickers, -1, -1)[:, perm, :, :]
    )
    future_perm_x = future_perm.reshape(B_sz * n_tickers, 1, n_future).to(device)

    with torch.no_grad():
        preds_perm = (
            model(static_perm_x, past_perm_x, future_perm_x).view(B_sz, n_tickers).cpu()
        )

    # Restore the original order from the permuted prediction
    preds_perm_restored = preds_perm[:, perm]

    diff = torch.abs(preds_orig - preds_perm_restored).max().item()
    print(f"  Max absolute discrepancy after permutation and restoration: {diff:.8f}")

    if diff < 1e-5:
        print("  [PASS] Ticker permutation invariance verified!")
    else:
        print(
            "  [FAIL] Ticker permutation invariance failed. Predictions differ when tickers are permuted."
        )
