#!/usr/bin/env python3
"""Export the selected PyTorch ANN and scalers to a lightweight NumPy artifact."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch


def json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--scalers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    scalers = joblib.load(args.scalers)
    state = checkpoint["state_dict"]
    layer_indices = sorted(
        int(key.split(".")[1])
        for key in state
        if key.endswith(".weight")
    )

    arrays = {
        "x_mean": np.asarray(scalers["x_scaler"].mean_, dtype=np.float64),
        "x_scale": np.asarray(scalers["x_scaler"].scale_, dtype=np.float64),
        "y_mean": np.asarray(scalers["y_scaler"].mean_, dtype=np.float64),
        "y_scale": np.asarray(scalers["y_scaler"].scale_, dtype=np.float64),
    }
    for output_index, layer_index in enumerate(layer_indices):
        arrays[f"weight_{output_index}"] = state[f"network.{layer_index}.weight"].numpy()
        arrays[f"bias_{output_index}"] = state[f"network.{layer_index}.bias"].numpy()

    metadata = {
        "feature_columns": checkpoint["feature_columns"],
        "target_column": checkpoint["target_column"],
        "hidden_layers": checkpoint["hidden_layers"],
        "activation": checkpoint["activation"],
        "metrics": checkpoint["metrics"],
        "n_layers": len(layer_indices),
        "source_model": str(args.model),
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, default=json_default),
        dtype=np.str_,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
