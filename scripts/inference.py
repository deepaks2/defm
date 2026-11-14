# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from defm import preprocess_depth_image
from defm.model_factory import create_defm_model


def resolve_device(name: str) -> torch.device:
    if name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU was selected, but no Intel XPU is available")
    elif name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but no CUDA device is available")
    return torch.device(name)


def summarize_output(output):
    if torch.is_tensor(output):
        return {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "device": str(output.device),
            "finite": bool(torch.isfinite(output).all()),
        }
    if isinstance(output, dict):
        return {key: summarize_output(value) for key, value in output.items()}
    if isinstance(output, (list, tuple)):
        return [summarize_output(value) for value in output]
    return type(output).__name__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeFM inference on one depth image")
    parser.add_argument("--model", default="defm_resnet18")
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="cpu")
    parser.add_argument("--input", type=Path, required=True, help="Depth image in meters")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(name in args.model for name in ("resnet", "regnet", "efficientnet")):
        if args.target_size % 32:
            raise ValueError("CNN models require --target-size to be a multiple of 32")
    device = resolve_device(args.device)
    model = create_defm_model(
        args.model,
        pretrained=not args.no_pretrained,
        pretrained_path=str(args.checkpoint) if args.checkpoint else None,
    ).eval().to(device)
    image = Image.open(args.input)
    inputs = preprocess_depth_image(
        image, target_size=args.target_size, patch_size=args.patch_size
    ).to(device)

    with torch.inference_mode():
        output = model(inputs)

    print(
        json.dumps(
            {
                "model": args.model,
                "device": str(device),
                "input": {
                    "shape": list(inputs.shape),
                    "dtype": str(inputs.dtype),
                    "device": str(inputs.device),
                },
                "output": summarize_output(output),
            }
        )
    )


if __name__ == "__main__":
    main()
