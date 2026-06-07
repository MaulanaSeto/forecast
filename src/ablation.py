"""
Ablation Study module for TFT on IDX Stock Forecasting.

Runs ablation experiments by training model variants with specific
components removed to measure their contribution to model performance.

Ablation 1: TFT tanpa Variable Selection Network (VSN)
Ablation 2: TFT tanpa Temporal Self-Attention
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .model import TFT, TFT_NoVSN, TFT_NoAttention
from .train import train_model

matplotlib.use("Agg")


def run_ablation_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_kwargs: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    save_dir: str = "checkpoints/ablation",
) -> dict[str, Any]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """
    Run full ablation study: train baseline + 2 ablated variants.

    Args:
        train_loader: training DataLoader
        val_loader: validation DataLoader
        model_kwargs: dict of kwargs for TFT constructor (n_static, n_past, n_future, etc.)
        config: training config dict (lr, weight_decay, max_epochs, patience, grad_clip)
        device: torch device
        save_dir: directory to save ablation results

    Returns:
        results: dict mapping experiment name -> {history, best_val_rmse}
    """
    os.makedirs(save_dir, exist_ok=True)

    experiments = {
        "TFT_Full": TFT,
        "TFT_NoVSN": TFT_NoVSN,
        "TFT_NoAttention": TFT_NoAttention,
    }

    results = {}

    for exp_name, ModelClass in experiments.items():
        # Create model
        model = ModelClass(**model_kwargs).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            save_dir=save_dir,
            model_name=exp_name.lower(),
        )

        results[exp_name] = {
            "history": history,
            "best_val_rmse": history["best_val_rmse"],
            "best_epoch": history["best_epoch"],
            "n_params": n_params,
        }
        rmse_val = history["best_val_rmse"]
        epoch_val = history["best_epoch"]

        import torch.distributed as dist

        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if is_main_process:
            print(f"  {exp_name:15} | RMSE: {rmse_val:.6f} | Epoch: {epoch_val}")

    # Summary
    if is_main_process:
        save_ablation_results(results, save_dir)

        # Generate ablation comparison plot
        plot_path = os.path.join(save_dir, "ablation_comparison.png")
        plot_ablation_comparison(results, plot_path)

    return results


def print_ablation_summary(results: dict[str, Any]) -> None:
    """Print a formatted summary table of ablation results."""
    print("\n--- ABLATION SUMMARY ---")
    baseline_rmse = results.get("TFT_Full", {}).get("best_val_rmse", None)

    for name, res in results.items():
        rmse = res["best_val_rmse"]
        params = res["n_params"]
        delta = ""
        if baseline_rmse is not None and name != "TFT_Full":
            pct = ((rmse - baseline_rmse) / baseline_rmse) * 100
            delta = f" ({pct:+.2f}%)"
        print(f"  {name:<16} | RMSE: {rmse:.6f}{delta} | Params: {params:,}")
    print("------------------------")


def save_ablation_results(results: dict[str, Any], save_dir: str) -> None:
    """Save ablation results to JSON file."""
    # Convert to JSON-serializable format
    json_results: dict[str, Any] = {}
    for name, res in results.items():
        json_results[name] = {
            "best_val_rmse": float(res["best_val_rmse"]),
            "best_epoch": int(res["best_epoch"]),
            "n_params": int(res["n_params"]),
            "train_loss_history": [float(x) for x in res["history"]["train_loss"]],
            "val_rmse_history": [float(x) for x in res["history"]["val_rmse"]],
        }

    json_results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    path = os.path.join(save_dir, "ablation_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Ablation results saved to {path}")


def plot_ablation_comparison(
    results: dict[str, Any], save_path: str = "ablation_comparison.png"
) -> None:
    # pylint: disable=too-many-local-variables
    """
    Plot side-by-side comparison of ablation experiments.

    Creates a figure with:
    1. Bar chart of best RMSE per model
    2. Training curves overlaid
    """
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    rmses = [results[n]["best_val_rmse"] for n in names]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    # Bar chart
    ax1 = axes[0]
    rect_bars = ax1.bar(
        names, rmses, color=colors[: len(names)], edgecolor="black", linewidth=0.5
    )
    ax1.set_ylabel("Best Validation RMSE")
    ax1.set_title("Ablation Study — Model Comparison")
    ax1.set_ylim(min(rmses) * 0.98, max(rmses) * 1.02)
    for rect, rmse in zip(rect_bars, rmses):
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{rmse:.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Training curves
    ax2 = axes[1]
    for i, name in enumerate(names):
        history = results[name]["history"]
        ax2.plot(history["val_rmse"], label=name, color=colors[i], linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation RMSE")
    ax2.set_title("Validation RMSE Over Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Ablation comparison plot saved to {save_path}")
