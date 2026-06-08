"""Training loop, evaluation, and experiment logging for the TFT model.

This module provides:
- ``train_epoch``  - one pass over the training set with gradient clipping.
- ``evaluate``     - inference pass that returns RMSE plus raw arrays.
- ``train_model``  - full training pipeline (scheduling, early stopping, checkpointing).
- ``plot_training_history`` - dual-axis plot of loss and validation RMSE.
- ``run_experiment`` - convenience wrapper that ties everything together.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl  # type: ignore
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

matplotlib.use("Agg")  # non-interactive backend so saving always works


# Training helpers
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float = 1.0,
) -> float:
    # pylint: disable=too-many-local-variables
    """Run a single training epoch."""
    import torch.distributed as dist

    is_distributed = dist.is_initialized()
    local_rank = dist.get_rank() if is_distributed else 0
    disable_tqdm = is_distributed and local_rank != 0

    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n_batches = 0

    progress = tqdm(loader, desc="  [Train]", leave=False, disable=disable_tqdm)
    for batch in progress:
        static_x = batch["static"].to(device, non_blocking=True)
        past_x = batch["past"].to(device, non_blocking=True)
        future_x = batch["future"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Enable AMP
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            predictions = model(static_x, past_x, future_x).squeeze(-1)
            loss = criterion(predictions, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        progress.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = total_loss / max(n_batches, 1)
    if is_distributed:
        avg_loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / dist.get_world_size()

    return avg_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model and return RMSE, MSE, and Directional Accuracy."""
    import torch.distributed as dist

    is_distributed = dist.is_initialized()
    local_rank = dist.get_rank() if is_distributed else 0
    disable_tqdm = is_distributed and local_rank != 0

    model.eval()
    preds_list: list[Tensor] = []
    targets_list: list[Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  [Eval] ", leave=False, disable=disable_tqdm):
            static_x = batch["static"].to(device, non_blocking=True)
            past_x = batch["past"].to(device, non_blocking=True)
            future_x = batch["future"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(static_x, past_x, future_x).squeeze(-1)
            preds_list.append(preds.cpu())
            targets_list.append(targets.cpu())

    all_preds = torch.cat(preds_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)

    mse = torch.mean((all_preds - all_targets) ** 2).item()
    correct_dir = (
        (torch.sign(all_preds) == torch.sign(all_targets)).float().mean().item()
    )

    if is_distributed:
        metrics = torch.tensor([mse, correct_dir], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= dist.get_world_size()
        mse = metrics[0].item()
        correct_dir = metrics[1].item()

    rmse = float(np.sqrt(mse))

    return rmse, mse, correct_dir


def save_history_csv(history: dict[str, list], save_dir: str) -> None:
    """Save training history to a CSV file for analysis in IPYNB."""
    df = pl.DataFrame(history)
    path = os.path.join(save_dir, "metrics.csv")
    df.write_csv(path)
    print(f"  History saved to {path}")


# Full training pipeline
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    save_dir: str = "checkpoints",
    model_name: str = "tft",
) -> dict[str, Any]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-local-variables
    """
    Train the TFT model using the provided data loaders and configuration.

    Args:
        model: The TFT model architecture
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Dictionary containing training hyperparameters (lr, weight_decay, etc.)
        device: Torch device (cpu or cuda)
        save_dir: Path to directory for saving model checkpoints
        model_name: Prefix for the saved model file

    Returns:
        history: Dictionary containing training and validation metrics per epoch
    """
    import torch.distributed as dist

    is_distributed = dist.is_initialized()
    local_rank = dist.get_rank() if is_distributed else 0
    is_main_process = not is_distributed or local_rank == 0

    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            bucket_cap_mb=15,
        )

    # Compile model for faster training on PyTorch 2.x
    if config.get("compile_model", False) and hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(model)
            if is_main_process:
                print(
                    "  [Warmup] Testing torch.compile() with a dry-run forward pass..."
                )

            # Fetch a dummy batch from loader
            dummy_batch = next(iter(train_loader))
            dummy_static = dummy_batch["static"][:2].to(device)
            dummy_past = dummy_batch["past"][:2].to(device)
            dummy_future = dummy_batch["future"][:2].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                with torch.no_grad():
                    _ = compiled_model(dummy_static, dummy_past, dummy_future)

            model = compiled_model
            if is_main_process:
                print("  [OK] torch.compile() enabled successfully.")
        except Exception as e:
            if is_main_process:
                print(
                    f"  [WARNING] torch.compile failed during warmup: {e}. Falling back to eager mode."
                )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Historical bookkeeping
    history: dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
        "train_dir_acc": [],
        "val_dir_acc": [],
        "lr": [],
    }

    best_val_rmse = float("inf")
    best_epoch, patience_counter = 0, 0
    max_epochs, patience = config["max_epochs"], config["patience"]
    grad_clip = config["grad_clip"]
    best_state_path = os.path.join(save_dir, f"{model_name}_best.pt")

    if is_main_process:
        print(
            f"  Training: {max_epochs} epochs | batch={train_loader.batch_size} | lr={config['lr']}"
        )

    best_state_dict = None

    for epoch in range(1, max_epochs + 1):
        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        curr_lr = optimizer.param_groups[0]["lr"]

        # --- train ---
        avg_train_mse = train_epoch(
            model, train_loader, optimizer, device, scaler, grad_clip
        )
        train_rmse = float(np.sqrt(avg_train_mse))

        # --- validate ---
        val_rmse, val_mse, val_dir = evaluate(model, val_loader, device)

        # --- logging & history ---
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_mse)
        history["val_loss"].append(val_mse)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["val_dir_acc"].append(val_dir)
        history["lr"].append(curr_lr)
        history["train_dir_acc"].append(0.0)  # Speed opt (skip train-dir eval)

        if is_main_process:
            msg = (
                f"  Epoch {epoch:02d}/{max_epochs:02d} | loss: {avg_train_mse:.6f} | "
                f"val_rmse: {val_rmse:.6f} | dir: {val_dir:.2%}"
            )
            print(msg)

        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            raw_model = model
            if hasattr(raw_model, "module"):
                raw_model = raw_model.module
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
            best_state_dict = {
                k: v.cpu().clone() for k, v in raw_model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if is_main_process:
                print(f"  Early stopping at epoch {epoch}")
            break

    # --- save best model ---
    if is_main_process:
        if best_state_dict is not None:
            torch.save(best_state_dict, best_state_path)
            print(
                f"\n  [OK] Best model (Epoch {best_epoch}) saved to {os.path.basename(best_state_path)}"
            )
            raw_model = model
            if hasattr(raw_model, "module"):
                raw_model = raw_model.module
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
            raw_model.load_state_dict(best_state_dict)
        else:
            raw_model = model
            if hasattr(raw_model, "module"):
                raw_model = raw_model.module
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
            torch.save(raw_model.state_dict(), best_state_path)
            print(f"\n  [OK] Final model saved to {os.path.basename(best_state_path)}")

        save_history_csv(history, save_dir)
    history["best_val_rmse"] = best_val_rmse
    history["best_epoch"] = best_epoch
    return history


# Plotting
def plot_training_history(
    history: dict[str, Any],
    save_path: str = "training_history.png",
) -> None:
    """Plot training loss and validation RMSE."""
    epochs = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_loss = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (MSE)", color=color_loss)
    ax1.plot(
        epochs,
        history["train_loss"],
        color=color_loss,
        marker="o",
        markersize=3,
        label="Train Loss",
    )
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    color_rmse = "tab:red"
    ax2.set_ylabel("Validation RMSE", color=color_rmse)
    ax2.plot(
        epochs,
        history["val_rmse"],
        color=color_rmse,
        marker="s",
        markersize=3,
        label="Val RMSE",
    )
    ax2.tick_params(axis="y", labelcolor=color_rmse)

    fig.suptitle("Training History", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# Experiment runner
def run_experiment(
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    experiment_name: str = "experiment",
) -> tuple[nn.Module, dict[str, Any]]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Convenience wrapper for experiments."""
    model = model_class(**model_kwargs)
    save_dir = os.path.join("checkpoints", experiment_name)
    history = train_model(
        model,
        train_loader,
        val_loader,
        config,
        device,
        save_dir=save_dir,
        model_name=experiment_name,
    )
    plot_training_history(
        history, os.path.join(save_dir, f"{experiment_name}_history.png")
    )
    return model, history
