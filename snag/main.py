import argparse
import json
import datetime
import numpy as np
import torch
from pathlib import Path

from .trainer import Trainer


def ensure_unique_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        if output_dir.is_dir() and any(output_dir.iterdir()):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir.with_name(f"{output_dir.name}_{timestamp}")
        elif output_dir.is_file():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir.with_name(f"{output_dir.name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_int_list(value):
    if value is None:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_float_list(value):
    if value is None:
        return None
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_str_list(value):
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def build_args():
    parser = argparse.ArgumentParser(description="SNAG GATv2 search")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "derive"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--fold_dir", type=str, default="data/folds_data")
    parser.add_argument("--fold_name", type=str, default="")
    parser.add_argument("--fold_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="results/snag_gatv2")

    parser.add_argument("--layers_of_child_model", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--normalize_method", type=str, default="standard")

    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--shared_initial_step", type=int, default=5)
    parser.add_argument("--shared_params", action="store_true", default=True)
    parser.add_argument("--no_shared_params", action="store_false", dest="shared_params")
    parser.add_argument("--controller_max_step", type=int, default=10)
    parser.add_argument("--controller_lr", type=float, default=3.5e-4)
    parser.add_argument("--controller_hid", type=int, default=100)
    parser.add_argument("--controller_optim", type=str, default="adam")
    parser.add_argument("--controller_grad_clip", type=float, default=0)
    parser.add_argument("--softmax_temperature", type=float, default=5.0)
    parser.add_argument("--tanh_c", type=float, default=2.5)

    parser.add_argument("--entropy_mode", type=str, default="reward")
    parser.add_argument("--entropy_coeff", type=float, default=1e-4)
    parser.add_argument("--ema_baseline_decay", type=float, default=0.95)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--time_budget", type=float, default=0.0)
    parser.add_argument("--derive_num_sample", type=int, default=10)

    parser.add_argument("--heads", type=str, default="")
    parser.add_argument("--hidden_dims", type=str, default="")
    parser.add_argument("--dropouts", type=str, default="")
    parser.add_argument("--activations", type=str, default="")
    parser.add_argument("--jk_modes", type=str, default="")

    return parser.parse_args()


def main():
    args = build_args()
    if args.device == "cpu":
        args.cuda = False
    elif args.device == "cuda":
        args.cuda = True
    else:
        args.cuda = True

    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    project_root = Path(__file__).resolve().parents[1]
    if args.fold_dir and not Path(args.fold_dir).is_absolute():
        args.fold_dir = str(project_root / args.fold_dir)
    if args.fold_path and not Path(args.fold_path).is_absolute():
        args.fold_path = str(project_root / args.fold_path)

    search_space = {}
    heads = parse_int_list(args.heads)
    hidden_dims = parse_int_list(args.hidden_dims)
    dropouts = parse_float_list(args.dropouts)
    activations = parse_str_list(args.activations)
    jk_modes = parse_str_list(args.jk_modes)

    if heads:
        search_space["heads"] = heads
    if hidden_dims:
        search_space["hidden_dim"] = hidden_dims
    if dropouts:
        search_space["dropout"] = dropouts
    if activations:
        search_space["activation"] = activations
    if jk_modes:
        search_space["jk_mode"] = jk_modes
    if search_space:
        args.search_space = search_space

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = ensure_unique_output_dir(output_dir)

    trainer = Trainer(args)
    if args.mode == "train":
        results = trainer.train()
    else:
        actions, score = trainer.derive()
        results = {
            "best_actions": actions,
            "best_val_score": float(score),
            "history": trainer.history,
        }

    model_state = None
    if results.get("finetune") and "model_state" in results["finetune"]:
        model_state = results["finetune"].pop("model_state")

    if model_state is not None:
        torch.save(model_state, output_dir / "snag_best_model.pt")

    with open(output_dir / "snag_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "snag_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
