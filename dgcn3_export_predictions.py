import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch


def _add_dgcn3_path() -> str:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    dgcn3_path = os.path.join(repo_root, "demo", "DGCN3")
    if dgcn3_path not in sys.path:
        sys.path.insert(0, dgcn3_path)  # Insert at beginning to override root main.py
    return dgcn3_path


def _parse_alpha_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _save_predictions(probs: List[torch.Tensor], timesteps: List[int], dataset, alpha: float, output_base_dir: str) -> None:
    alpha_dir = os.path.join(output_base_dir, f"alpha_{alpha}")
    os.makedirs(alpha_dir, exist_ok=True)
    for pred, t in zip(probs, timesteps):
        mask = dataset[t].node_mask.cpu().numpy()
        node_ids = np.where(mask)[0]
        predictions = pred.numpy()[mask]
        df = pd.DataFrame({"node_id": node_ids, "centrality": predictions})
        output_path = os.path.join(alpha_dir, f"tiage_{t}.csv")
        df.to_csv(output_path, index=False, header=False)
        print(f"[*] Saved centrality predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="tiage")
    parser.add_argument("--alphas", type=str, default="1.5", help="comma-separated alpha list")
    parser.add_argument("--model_path", type=str, default="", help="override model path")
    parser.add_argument("--output_dir", type=str, default="", help="override output directory")
    args = parser.parse_args()

    dgcn3_path = _add_dgcn3_path()

    # Change to DGCN3 directory to load config.ini correctly
    original_cwd = os.getcwd()
    os.chdir(dgcn3_path)

    from data import get_data
    from model import NodeImportanceModel
    from main import load_hparams

    hparams = load_hparams()

    model_dir = os.path.join(dgcn3_path, "model_registry")
    os.makedirs(model_dir, exist_ok=True)
    model_path = args.model_path or os.path.join(model_dir, f"node_importance_{args.dataset_name}.pkl")

    output_dir = args.output_dir or os.path.join(dgcn3_path, "Centrality")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device selected: {device}")
    print(f"[*] Using model path: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    alpha_list = _parse_alpha_list(args.alphas)
    for alpha in alpha_list:
        # Stay in DGCN3 directory for data loading
        dataset, _, _ = get_data(
            dataset_name=args.dataset_name,
            train_test_ratio=hparams["train_test_ratio"],
            device=device,
            a=alpha
        )
        input_dim = dataset[0].x.size(1)
        model = NodeImportanceModel(
            input_dim=input_dim,
            hidden_dim=hparams["hidden_dim"],
            output_dim=hparams["output_dim"]
        ).to(device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        timesteps = list(range(len(dataset)))
        preds = []
        for t in timesteps:
            print(f"[*] Predicting timestep {t} for alpha={alpha}...")
            with torch.no_grad():
                y_pred = model.predict(graph=dataset[t], normalize=True)
            preds.append(y_pred.cpu())

        _save_predictions(preds, timesteps, dataset, alpha, output_dir)

    # Change back to original directory
    os.chdir(original_cwd)


if __name__ == "__main__":
    main()
