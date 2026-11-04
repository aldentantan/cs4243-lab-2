"""Stage 4 MiniMax-H3 Ref2VA inference for an A100-80GB or larger GPU.

Uses component-level automatic CPU offloading and the original interval video
plus artwork. No masks, gray filling, classical overlay, prompt expansion,
book tracking or board repair is performed. [AI-CODE]
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path.home() / "minimax_h3_stage4",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    model = root / "models/MiniMax-H3"
    video = root / "inputs/interval.mp4"
    artwork = root / "inputs/cliff.png"
    out_dir = args.out.expanduser().resolve()

    for path in [model / "modular_model_index.json", video, artwork]:
        if not path.is_file():
            raise FileNotFoundError(path)

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "output.mp4"
    request_path = out_dir / "request.json"
    if output.exists() or request_path.exists():
        raise FileExistsError(
            "Preserving previous attempt; use a new output directory."
        )

    import av
    import torch
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )
    from diffusers.utils.export_utils import encode_video

    if not torch.cuda.is_available():
        raise RuntimeError("Run this script inside a Slurm GPU allocation.")
    if "libx264" not in av.codecs_available:
        raise RuntimeError("This PyAV installation lacks the libx264 encoder.")
    torch.cuda.set_device(0)

    prompt = (
        "Replace the advertisement on the board in <Video 1> "
        "with the attached artwork <Picture 1>."
    )
    download_metadata = {
        str(path.relative_to(model)): path.read_text()
        for path in (model / ".cache/huggingface/download").rglob("*.metadata")
    }
    request = {
        "model": "MiniMaxAI/MiniMax-H3",
        "model_directory": str(model),
        "workflow": "ref2va",
        "prompt": prompt,
        "seed": 500,
        "dtype": "bfloat16",
        "quantization": None,
        "num_inference_steps": 30,
        "num_frames": 124,
        "output_fps": 24,
        "height": 480,
        "width": 864,
        "reference_image_short_edge": 768,
        "source_interval_inclusive": [120, 269],
        "source_fps": 30,
        "audit_frames": {
            "clean": 228, "occluded": 258,
            "perspective": 137, "pair": [246, 247],
        },
        "references_in_order": [str(video), str(artwork)],
        "input_sha256": {
            "video": sha256(video), "artwork": sha256(artwork),
        },
        "helper_sha256": sha256(Path(__file__)),
        "model_index_sha256": sha256(model / "modular_model_index.json"),
        "download_metadata": download_metadata,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "torch", "torchvision", "diffusers", "transformers",
                "accelerate", "av",
            ]
        },
        "offloading": "Component-level automatic CPU offload; 12GB GPU reserve",
        "declaration": (
            "[AI-GENERATION] Separate H3 experiment; preserve earlier attempts."
        ),
    }
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    print(json.dumps(request, indent=2), flush=True)

    started = time.perf_counter()
    print("Loading local Ref2VA components into CPU memory...", flush=True)
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(
        str(model),
        workflow="ref2va",
        components_manager=manager,
        local_files_only=True,
    )
    # Override Hub paths embedded in the index to use the downloaded directory.
    pipe.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=str(model),
        local_files_only=True,
    )
    if pipe.null_component_names:
        raise RuntimeError(
            f"Components failed to load: {pipe.null_component_names}"
        )

    pipe.update_components(
        canvas_short_edge=480,
        canvas_max_pixels=480 * 864,
        reference_image_short_edge=768,
    )
    pipe.transformer_ref.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Move complete components between stages. Keep the denoiser on the GPU
    # during sampling instead of streaming its individual blocks every step.
    manager.enable_auto_cpu_offload(
        device="cuda:0",
        memory_reserve_margin="12GB",
    )

    print("Decoding the original interval and artwork references...", flush=True)
    references = [
        MiniMaxH3VideoReference.from_file(str(video)),
        MiniMaxH3ImageReference.from_file(str(artwork)),
    ]
    if (abs(float(references[0].fps) - 30) > 1e-3
            or len(references[0].frames) != 150):
        raise ValueError("Expected the original 150-frame interval at 30 fps.")

    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    print("Encoding references, then running sampling...", flush=True)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            references=references,
            height=480,
            width=864,
            num_frames=124,
            num_inference_steps=30,
            generator=torch.Generator(device="cuda").manual_seed(500),
            output=["videos", "audio", "sampling_rate"],
        )
    generation_seconds = time.perf_counter() - generation_started

    print("Saving generated video and audio...", flush=True)
    encode_video(
        result["videos"][0],
        fps=24,
        output_path=str(output),
        audio=result["audio"][0],
        audio_sample_rate=result["sampling_rate"],
    )
    summary = {
        "output": str(output),
        "output_sha256": sha256(output),
        "actual_frames": len(result["videos"][0]),
        "output_fps": 24,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
