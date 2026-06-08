"""
Main entry point for IDX Stock Forecasting with Temporal Fusion Transformer (TFT).

Pipeline:
    1. Preprocess data
    2. Create datasets & dataloaders
    3. Train TFT model
    4. Evaluate and visualize
    5. Run inference on test set
    6. Generate submission CSV
    7. Run ablation study

Usage:
    python main.py                   # Full pipeline
    python main.py --mode train      # Train only
    python main.py --mode inference  # Inference only (requires trained model)
    python main.py --mode ablation   # Run ablation study
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Any

import yaml  # type: ignore
import torch

from src.preprocessing import preprocess_data
from src.dataset import create_dataloaders
from src.model import TFT
from src.train import train_model, plot_training_history
from src.inference import run_inference
from src.ablation import run_ablation_study


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from a YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        # Fallback defaults if YAML is missing
        "d_model": 64,
        "n_heads": 4,
        "n_lstm_layers": 2,
        "dropout": 0.1,
        "lookback": 60,
        "horizon": 1,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 30,
        "patience": 5,
        "grad_clip": 1.0,
        "val_split": 0.8,
        "data_dir": "dataset",
        "save_dir": "checkpoints",
        "submission_path": "submission.csv",
        "num_workers": 0,
        "max_gpus": 1,
        "use_checkpoint": True,
    }


def setup_device(config: dict[str, Any] | None = None) -> torch.device:
    """Setup and return the best available device (CUDA/CPU)."""
    local_rank = config.get("local_rank", 0) if config else 0
    is_distributed = config.get("is_distributed", False) if config else False

    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device("cuda", local_rank)
            name = torch.cuda.get_device_name(local_rank)
        else:
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        name = "CPU"

    if not is_distributed or local_rank == 0:
        print(f"  Device: {device} ({name})")
    return device


def run_training_pipeline(
    config: dict[str, Any],
) -> tuple[TFT, dict[str, Any], dict[str, Any], dict[str, Any]]:
    local_rank = config.get("local_rank", 0)
    is_main_process = not config.get("is_distributed", False) or local_rank == 0

    if is_main_process:
        print("\n>>> STARTING TRAINING PIPELINE")
    device = setup_device(config)

    # Preprocess
    if config.get("debug", False):
        config.update(
            {
                "lookback": config.get("debug_lookback", 10),
                "batch_size": config.get("debug_batch_size", 16),
                "max_epochs": config.get("debug_max_epochs", 2),
                "patience": config.get("debug_max_epochs", 2),
            }
        )

    data_dict = preprocess_data(config["data_dir"], config["lookback"])
    if config.get("debug", False):
        data_dict = make_debug_subset(data_dict, config)

    # Dataloaders
    loaders = create_dataloaders(
        data_dict,
        config["lookback"],
        config["batch_size"],
        config["val_split"],
        config.get("num_workers", 0),
        config.get("is_distributed", False),
    )

    # Model
    model_kwargs = {
        "n_static": data_dict["static_matrix"].shape[1],
        "n_past": len(data_dict["past_cols"]),
        "n_future": len(data_dict["future_cols"]),
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "n_lstm_layers": config["n_lstm_layers"],
        "dropout": config["dropout"],
        "lookback": config["lookback"],
        "horizon": config["horizon"],
        "use_checkpoint": config.get("use_checkpoint", True),
    }
    model = TFT(**model_kwargs).to(device)

    # Check if a logical debug mode is specified
    debug_mode = config.get("debug_mode", "none")
    if debug_mode in ("overfit", "causality", "permutation"):
        if is_main_process:
            from src.debug import (
                run_overfit_test,
                run_causality_test,
                run_permutation_test,
            )

            if debug_mode == "overfit":
                run_overfit_test(model, loaders["train_loader"], device, config)
            elif debug_mode == "causality":
                run_causality_test(model, loaders["val_loader"], device)
            elif debug_mode == "permutation":
                run_permutation_test(model, loaders["val_loader"], device)
        return model, {}, data_dict, loaders

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        print(f"  Model: TFT ({n_params:,} params)")

    # Train
    history = train_model(
        model=model,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        config=config,
        device=device,
        save_dir=config["save_dir"],
        model_name="tft_full",
    )

    # Plot & Inference (Only on main process / rank 0)
    if is_main_process:
        plot_training_history(history, os.path.join(config["save_dir"], "history.png"))
        best_model_path = os.path.join(config["save_dir"], "tft_full_best.pt")
        model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        model = model.to(device)

        run_inference(model, data_dict, config, device, config["submission_path"])

        msg = (
            f"\n>>> PIPELINE COMPLETE | Best Val RMSE: {history['best_val_rmse']:.6f} | "
            f"Session: {os.path.basename(config['save_dir'])}"
        )
        print(msg)

        # Save config for reproducibility
        config_path = os.path.join(config["save_dir"], "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    return model, history, data_dict, loaders


def run_inference_only(config: dict[str, Any]) -> None:
    """Run inference using a pre-trained model."""
    print("\n>>> STARTING INFERENCE ONLY")
    device = setup_device(config)

    data_dict = preprocess_data(config["data_dir"], config["lookback"])
    if config.get("debug", False):
        data_dict = make_debug_subset(data_dict, config)

    model_kwargs = {
        "n_static": data_dict["static_matrix"].shape[1],
        "n_past": len(data_dict["past_cols"]),
        "n_future": len(data_dict["future_cols"]),
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "n_lstm_layers": config["n_lstm_layers"],
        "dropout": config["dropout"],
        "lookback": config["lookback"],
        "horizon": config["horizon"],
        "use_checkpoint": config.get("use_checkpoint", True),
    }
    model = TFT(**model_kwargs).to(device)

    best_model_path = os.path.join(config["save_dir"], "tft_full_best.pt")
    if not os.path.exists(best_model_path):
        print(f"  ERROR: Model not found at {best_model_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    run_inference(model, data_dict, config, device, config["submission_path"])
    print(f">>> INFERENCE COMPLETE | Output: {config['submission_path']}")


def run_ablation(config: dict[str, Any]) -> dict[str, Any]:
    """Run the ablation study."""
    local_rank = config.get("local_rank", 0)
    is_main_process = not config.get("is_distributed", False) or local_rank == 0

    if is_main_process:
        print("\n>>> STARTING ABLATION STUDY")
    device = setup_device(config)

    # Preprocess
    if config.get("debug", False):
        config.update(
            {
                "lookback": config.get("debug_lookback", 10),
                "batch_size": config.get("debug_batch_size", 16),
                "max_epochs": config.get("debug_max_epochs", 2),
                "patience": config.get("debug_max_epochs", 2),
            }
        )

    data_dict = preprocess_data(config["data_dir"], config["lookback"])
    if config.get("debug", False):
        data_dict = make_debug_subset(data_dict, config)

    loaders = create_dataloaders(
        data_dict,
        config["lookback"],
        config["batch_size"],
        config["val_split"],
        config.get("num_workers", 0),
        config.get("is_distributed", False),
    )

    model_kwargs = {
        "n_static": data_dict["static_matrix"].shape[1],
        "n_past": len(data_dict["past_cols"]),
        "n_future": len(data_dict["future_cols"]),
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "n_lstm_layers": config["n_lstm_layers"],
        "dropout": config["dropout"],
        "lookback": config["lookback"],
        "horizon": config["horizon"],
        "use_checkpoint": config.get("use_checkpoint", True),
    }

    results = run_ablation_study(
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        model_kwargs=model_kwargs,
        config=config,
        device=device,
        save_dir=os.path.join(config["save_dir"], "ablation"),
    )

    local_rank = config.get("local_rank", 0)
    is_main_process = not config.get("is_distributed", False) or local_rank == 0
    if is_main_process:
        print("\n>>> ABLATION STUDY COMPLETE")
    return results


def make_debug_subset(
    data_dict: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """
    Slice data_dict arrays to make training and inference
    extremely fast for debugging.
    """
    subset_dict = data_dict.copy()
    n_train_total, n_test_total = (
        data_dict["train_past"].shape[0],
        data_dict["test_past"].shape[0],
    )
    pct = config.get("debug_data_pct", 0.01)

    train_size = max(int(n_train_total * pct), config["lookback"] * 5)
    test_size = max(int(n_test_total * pct), 10)

    subset_dict["train_past"] = data_dict["train_past"][-train_size:]
    subset_dict["train_future"] = data_dict["train_future"][-train_size:]
    subset_dict["train_targets"] = data_dict["train_targets"][-train_size:, :2]
    subset_dict["test_past"] = data_dict["test_past"][:test_size]
    subset_dict["test_future"] = data_dict["test_future"][:test_size]
    subset_dict["test_timestamps"] = data_dict["test_timestamps"][:test_size]
    subset_dict["static_matrix"] = data_dict["static_matrix"][:2]
    subset_dict["target_tickers"] = data_dict["target_tickers"][:2]
    subset_dict["n_train"], subset_dict["n_test"] = train_size, test_size

    local_rank = config.get("local_rank", 0)
    is_main_process = not config.get("is_distributed", False) or local_rank == 0
    if is_main_process:
        print(
            f"  [DEBUG] Subset created: {train_size} train, {test_size} test, 2 tickers"
        )
    return subset_dict


def main() -> None:
    """Entry point for the IDX Stock Forecasting pipeline."""
    parser = argparse.ArgumentParser(
        description="IDX Stock Forecasting with TFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Use default config.yaml
  python main.py --config my_cfg.yaml # Use custom config
  python main.py --mode train --epochs 50 # Override config via CLI
        """,
    )

    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "train", "inference", "ablation"],
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Path to dataset directory"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Path to save checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--lookback", type=int, default=None, help="Lookback window size"
    )
    parser.add_argument(
        "--d-model", type=int, default=None, help="Model hidden dimension"
    )
    parser.add_argument(
        "--n-heads", type=int, default=None, help="Number of attention heads"
    )
    parser.add_argument(
        "--patience", type=int, default=None, help="Early stopping patience"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Submission output path"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--max-gpus", type=int, default=None, help="Maximum number of GPUs to use"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with subsetted data for quick end-to-end verification",
    )
    parser.add_argument(
        "--debug-mode",
        type=str,
        default="none",
        choices=["none", "overfit", "causality", "permutation", "anomaly"],
        help="Logical debugging mode (default: none)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable gradient checkpointing in TFT's Variable Selection Networks",
    )

    args = parser.parse_args()

    # 1. Load config from YAML
    config = load_config(args.config)

    # 2. Override with CLI args if provided
    cli_map = {
        "data_dir": args.data_dir,
        "save_dir": args.save_dir,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lookback": args.lookback,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "patience": args.patience,
        "submission_path": args.output,
        "num_workers": args.workers,
        "max_gpus": args.max_gpus,
        "debug_mode": args.debug_mode,
    }

    for k, v in cli_map.items():
        if v is not None:
            config[k] = v

    config["debug"] = args.debug

    if args.no_checkpoint:
        config["use_checkpoint"] = False

    if config.get("debug_mode") == "anomaly":
        print(">>> PyTorch autograd anomaly detection ENABLED.")
        torch.autograd.set_detect_anomaly(True)

    # 3. Create dynamic session directory (only for training and ablation modes)
    if args.mode in ("full", "train", "ablation"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"run_{timestamp}"
        if config["debug"]:
            session_name = f"debug_{timestamp}"

        # Update save_dir to include timestamp
        config["save_dir"] = os.path.join(config["save_dir"], session_name)

    session_name = os.path.basename(config["save_dir"])

    # Create directories
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    # Update submission path to results folder
    config["submission_path"] = os.path.join(
        config["results_dir"], config["submission_path"]
    )

    msg = (
        f"\n>>> CONFIG: mode={args.mode} | lookback={config['lookback']} | "
        f"d_model={config['d_model']} | session={session_name}"
    )
    # Only print on master process
    if not config.get("is_distributed", False) or config.get("local_rank", 0) == 0:
        print(msg)

    # Run selected mode
    max_gpus = config.get("max_gpus", 1)
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_gpus = min(available_gpus, max_gpus)

    if num_gpus > 1 and args.mode in ("full", "train", "ablation"):
        import torch.multiprocessing as mp

        mp.spawn(
            ddp_worker, args=(num_gpus, config, args.mode), nprocs=num_gpus, join=True
        )
    else:
        if args.mode in ("full", "train"):
            run_training_pipeline(config)
        elif args.mode == "inference":
            run_inference_only(config)
        elif args.mode == "ablation":
            run_ablation(config)


def ddp_worker(
    local_rank: int, world_size: int, config: dict[str, Any], mode: str
) -> None:
    """Worker function for PyTorch DistributedDataParallel (DDP) spawned processes."""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    config_worker = config.copy()
    config_worker["local_rank"] = local_rank
    config_worker["world_size"] = world_size
    config_worker["is_distributed"] = True

    try:
        if mode in ("full", "train"):
            run_training_pipeline(config_worker)
        elif mode == "ablation":
            run_ablation(config_worker)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
