# sam2vit

*Center-focused* background removal with SAM 2 and prompt-based classification with CLIP.

This project segments the primary subject near the image center using SAM 2, paints the background white, then classifies each image using OpenAI CLIP with a structured prompt set. Images are saved to output subfolders named after the predicted class/item.

Example dataset (cars): https://www.kaggle.com/datasets/kshitij192/cars-image-dataset

## Key features

- SAM 2 center-focused segmentation (center point + margin refinement)
- Background painted white; output as RGBA
- Prompt builder (prefix × items × postfix) for dense CLIP scoring
- CLIP-based top-prompt selection and result reorganization
- Device-aware execution: CUDA, Apple MPS, or CPU (MPS CPU fallback supported)

## How it works (pipeline)

Input image → SAM 2 segmentation (center seed + margin refinement) → Background painted white (RGBA) → Prompt set built from prefixes/items/postfixes → CLIP scores all prompts → Best prompt selected → Image saved to output/<item>/

## Requirements

- Python 3.10+
- PyTorch (match your CUDA/MPS/CPU environment)
- OpenAI CLIP
- Optional for visualization only: matplotlib, opencv-python

## Installation

1) Install PyTorch suitable for your OS and GPU drivers (CUDA/MPS/CPU).
2) Install project dependencies and CLIP.
3) Ensure the SAM 2 package and configs exist under `sam2/configs/sam2.1/`.
4) Download the SAM 2 checkpoint into `./checkpoints/` (e.g., `sam2.1_hiera_large.pt`).

Example (adjust versions for your system):

```bash
pip install -e .
pip install numpy pillow rich
pip install git+https://github.com/openai/CLIP.git
# Install torch / torchvision matching your CUDA or CPU. See https://pytorch.org/get-started/locally/
```

## Usage

Prepare inputs in a folder (e.g., `dataset_cars/`) and run:

```bash
python main.py --dataset-dir dataset_cars --output-dir output
```

Useful flags:

- `--debug` / `-d`: verbose logging
- `--no-log`: disable file logging
- `--no-sam` / `-x`: skip SAM 2 (use original image for CLIP)
- `--force-device {cuda|cpu|mps}`: override device choice
- `--clip-model`: CLIP variant (default: `ViT-L/14@336px`)
- `--sam-model` / `--sam-config`: SAM 2 model and config names

The program prints device info and prompt counts. Results save to `output/<item>/<image>.png`.

## Configuration

- `config.json` contains three arrays: `prefixes`, `items`, and `postfixes`.
- These are combined to form prompts: `prefix + item + postfix`.
- See `main.py` for how they’re loaded; see `sam_utilities.py` for SAM 2 behavior.

Defaults worth noting:

- Focalization margin: `mask_threshold = 0.05` (5% of shorter image edge)
- Central point: fixed at the image center; two additional points offset by margin
- Background: painted white; image returned as RGBA

## Customization

- Change `items`, `prefixes`, and `postfixes` in `config.json` to adapt to other domains (e.g., apparel, furniture, produce).
- Switch to a smaller CLIP model via `--clip-model` (or `-c`): e.g., `ViT-B/32`) for speed.
- Adjust `mask_threshold` (in `main.py` call to `predict_sam2`) to widen/tighten the focal region.
- Modify `predict_sam2()` to place different points or to write the mask into the alpha channel for true transparency.

## Edge cases and behaviors

Image characteristics
- Very small images: the margin-based refinement can collapse to overlapping points; consider lowering `mask_threshold` or skipping refinement.
- Extremely large images: may cause OOM on limited GPUs; resize inputs or switch to a smaller CLIP model via `--clip-model` (or `-c`).
- Non-RGB inputs (grayscale/CMYK): converted to RGB internally; colors may shift slightly.
- Low contrast foreground/background: SAM 2 may segment poorly; we should adjust margin or provide custom prompts/points.

File handling
- Output extension is `.png`; inputs with other extensions will be saved as PNG.

Prompting and CLIP
- Prompt collisions: if different prompts tokenize similarly, probabilities may be close; consider diversifying prefixes/postfixes.
- Long prompts: CLIP tokenizes up to a maximum length (model-dependent, often 77 tokens); very long strings will be truncated.
- Class leakage: prefixes/postfixes should maintain spacing so items remain distinct tokens.

Devices and performance
- Device selection: SAM 2 can run on CUDA, MPS, or CPU; CLIP on CUDA or CPU. MPS CPU fallback is enabled via `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Mixed devices: SAM 2 and CLIP can run on different devices; see `main.py` for selection and `cuda_device.py` for checks.
- Precision/TF32: On CUDA, BF16 autocast and TF32 (Ampere+) can be enabled for speed; see `cuda_device.py`.

Dependencies
- Optional: `matplotlib` and `opencv-python` are only needed for visualization utilities; core flow runs without them.

Operational scenarios
- `--no-sam`: classification runs on the original image; useful to compare CLIP-only vs SAM+CLIP.
- Empty prompt lists: ensure `config.json` arrays are non-empty; otherwise CLIP will get an empty set.
- Missing checkpoints/configs: verify files exist at paths built by `get_checkpoint_path()` and `get_model_cfg_path()`.

## Troubleshooting

- CUDA not available / out of memory
  - Reduce batch size (images are processed singly here), resize inputs, or switch to CPU/MPS.
  - Use a smaller CLIP model.

- CLIP not installed
  - Install via `pip install git+https://github.com/openai/CLIP.git`.

- SAM 2 config/checkpoint not found
  - Ensure `sam2/configs/sam2.1/<name>.yaml` and `checkpoints/<model>.pt` exist.

- No images processed
  - Confirm `--dataset-dir` path and supported extensions. Check logs with `--debug`.

## Project structure (simplified)

- `main.py` — entry point; args, devices, model loading, loop
- `config.json` — prompt pieces (prefixes/items/postfixes)
- `clip_utilities.py` — CLIP wrapper (`ClipModel.process`)
- `sam_utilities.py` — SAM 2 helpers (`predict_sam2`)
- `sam2/` — SAM 2 package and configs (`sam2/configs/sam2.1/`)
- `checkpoints/` — SAM 2 checkpoints (e.g., `sam2.1_hiera_large.pt`)

## Roadmap

Done
- [x] Center-focused segmentation via SAM 2 (center seed + margin refinement)
- [x] Background painted white; RGBA output
- [x] Prompt builder (prefix × items × postfix) and CLIP scoring pipeline
- [x] Device-aware execution (CUDA, MPS, CPU) with MPS CPU fallback
- [x] Config-driven prompts via `config.json`
- [x] Optional no-SAM flow (`--no-sam`) for CLIP-only classification
- [x] Debug logging and console table output
- [x] Edge cases and troubleshooting documentation

Planned
- [ ] Optional object detector (e.g., YOLO) to set focal points adaptively
- [ ] Pluggable background handling (white vs. true transparency)
- [ ] Configurable prompt templates and tokenization limits
- [ ] Central position may be changed in the future as it's hardcoded.

## Acknowledgements

- SAM 2: https://github.com/facebookresearch/sam2/
- CLIP: https://github.com/openai/CLIP

## License

Apache License 2.0. See `LICENSE` for more details.
