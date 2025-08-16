# sam2vit

## What is it?

Highly precise and reliable datasets.

## SAM2 Center-Focus Background Removal + CLIP

This project segments the central subject of each image using SAM 2, removes the background, and classifies the vehicle's color using CLIP with a structured prompt workflow. Prompts are built from prefix prompts + color names + postfix prompts, merged into a single array, scored by CLIP, and the resulting images are reorganized into color-named folders.

**Case scenario dataset:** https://www.kaggle.com/datasets/kshitij192/cars-image-dataset
## TODO List
 - [x] Input Image --> SAM2 --> Get the `|prefix|x|item|x|postfix|` (**built prompt**) --> CLIP
 - [ ] Use `yolo` *(or another object detection algorithm)* to obtain the focal point better
 - [ ] Remove the back project (SAM2) and use requirements instead

### Features:

- Center-focused segmentation with SAM 2:

  - Uses the image center as the primary point with an additional 5% optional margin to refine the mask.

- Background removal:

  - Non-foreground pixels are set to white and the image is returned as RGBA.

- Prompt workflow:

  - Cartesian product of prefix prompts + color names + postfix prompts to form a dense prompt set.

- CLIP-based color classification:

  - Scores all prompts for each image and selects the top one.

- Result reorganization:

  - Saves segmented images into output/<predicted_color>/ directories.

- Device-aware:

  - SAM 2 on MPS (Apple), CUDA, or CPU; CLIP on CUDA or CPU.

  - MPS CPU fallback enabled automatically.



### Installation:

**Prerequisites:**

- Python 3.10
- CUDA 12.6


Example installation steps:

> Install **PyTorch** (match your system/CUDA):
> 
> Developed using **CUDA 12.6**

**Install commands:**

```python
pip install numpy pillow opencv-python matplotlib rich
pip install -e .
pip install git+https://github.com/openai/CLIP.git
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchmetrics==1.8.1 --extra-index-url https://download.pytorch.org/whl/cu126
```

- Make the SAM 2 package available:

  - Ensure a *sam2/configs/[model].yaml* directory exists in this repository or download from

- Download SAM 2 checkpoint:

  - Place the checkpoint file at `./checkpoints/sam2.1_hiera_large.pt` (or any other model or model name under `./checkpoint`)


## Customization:

- Prompts:

- Prompt size:

  - get_prompts() produces |prefix| × |items| × |postfix| combinations. Adjust lists to balance accuracy and speed.

- CLIP model:

  - You can change your clip_model_name (default "ViT-L/14@336px") to a smaller model for speed (e.g., "ViT-B/32") if desired.

- SAM 2 model/config:

  - You can change the sam_model_name and sam_config_name; update checkpoint and config paths accordingly, or disable SAM  via `-x`

- Focalization margin:

  - mask_threshold controls the 5% margin. Default call uses 0.05; modify the call to predict_sam2() in process() to change it (e.g., 0.03 for 3%).

- Center point:

  - predict_sam2() currently fixes the central point. To change focalization, alter point_coords in predict_sam2() to your desired coordinates.

- Input/output directories:

  - Use `--dataset-dir [input dir] --output [output dir]` 



## Expected Runtime and Memory:

- The default CLIP model (ViT-L/14@336px) is large; expect higher memory usage and longer runtimes per image compared to smaller variants.

- SAM 2 inference can be GPU-accelerated via MPS/CUDA; CPU-only runs will be slower.



## Notes and Limitations:

- Background handling:

  - By default, background pixels are set to white. If you need actual transparency, modify predict_sam2() to write the mask into the alpha channel instead of painting white.

- Color extraction:

  - get_color() searches for the color as a space-delimited token in the winning prompt. Ensure your prefix/postfix strings include spaces around color tokens or adjust get_color() accordingly.

- File extensions:

  - Output filenames are created by replacing ".jpg" with ".png". If your input images have other extensions, adjust the saving logic for consistency.



Reference Commands:

- Install CLIP:

  - pip install git+https://github.com/openai/CLIP.git

- Run in debug mode:

  - python main.py --debug

- Standard run:

  - python main.py



Project Structure [simplified]:

- main.py

- clip_utilities.py

- sam_utilities.py

- sam2/ [SAM 2 package with configs under sam2/configs/sam2.1/]

- checkpoints/

  - sam2.1_hiera_large.pt

- dataset_cars/

- output/



Contact/Attribution:

- Author: https://github.com/iamdroppy

- Contact: https://github.com/iamdroppy/sam2vit

- SAM 2: https://github.com/facebookresearch/sam2/

- CLIP: https://github.com/openai/CLIP

## Usage

Prepare your inputs by placing images in dataset_cars/. PIL-readable formats (e.g., .jpg, .png) are supported. Run the program with `python main.py`. For verbose logging, add `--debug` (or `-d`).

- Basic run: `python main.py`  
- Debug mode: `python main.py --debug`

Output images will be saved to output/<color>/<image_name>.png based on CLIP’s best-matching prompt color. For example, after running on `dataset_cars/car_001.jpg` and `dataset_cars/car_002.jpg` in debug mode, you might see `output/red/car_001.png` and `output/blue/car_002.png`.

## Inputs and Outputs

Inputs arrive from the dataset_cars/ directory and can be any format PIL supports. Internally, the SAM 2 configuration file is read from sam2/configs/sam2.1/sam2.1_hiera_l.yaml, and the checkpoint from checkpoints/sam2.1_hiera_large.pt.

Outputs are organized by predicted color in output/<color>/. Files are saved as PNG. The program prints device selections, prompt counts, and per-image probabilities to the console.

Prompt arrays are created as the Cartesian product of:
- 9 prefix prompts
- 18 colors
- 6 postfix prompts

for a total of 972 text prompts.

## End-to-end flow

Below is a compact view of the pipeline:

Input image → Center-focused SAM 2 segmentation (center + 5% margin points) → Background painted white (RGBA) → Prompts are generated (prefix + color + postfix) and merged → CLIP scores all prompts → Top prompt/color is extracted → Image saved in output/<color>/

## Configuration and defaults

Most behaviors are controlled in the repository’s Python files. Key settings and where to change them:

| Setting | Default | Where | Notes |
|---|---|---|---|
| Input directory | dataset_cars | main.py (process) | Change to process different datasets |
| Output directory | output/<color> | main.py (process) | Created automatically per color |
| Seed | 3 | main.py | For reproducibility of any stochastic ops |
| SAM 2 model | sam2.1_hiera_large | main.py | Checkpoint path is auto-built |
| SAM 2 config | sam2.1_hiera_l | main.py | Config path is auto-built |
| CLIP model | ViT-L/14@336px | main.py | Consider smaller models for speed |
| Focalization margin | 0.05 | sam_utilities.predict_sam2 | 5% of the shorter image edge |
| MPS fallback | Disabled by default | main.py (env) | `PYTORCH_ENABLE_MPS_FALLBACK=1` |

Notes on background handling and transparency: the output is RGBA, but the background is painted white. If you require true transparency, modify `predict_sam2()` in `sam_utilities.py` to write the mask into the alpha channel.

## Technical overview

- **SAM 2 focalization**

  An initial prediction is made from a central positive point. Masks are sorted by score; the highest-scoring logits are used for a second pass with three positive points: the center and two additional points offset using the 5% margin. The combined mask is then applied to the image, painting non-foreground pixels white.

- **Prompt generation and merging**

  `get_prompts(colors, prefix_prompts, postfix_prompts)` computes the Cartesian product. The resulting list is passed to CLIP as a single merged array.

- **CLIP execution** 

  The image is preprocessed and encoded; prompts are tokenized and encoded. Probabilities (softmax over logits) are computed and the top prompt is returned with its probability.

- **Reorganization**  
  The code extracts the color token from the top prompt and writes the processed PNG to `output/<color>/`. If the directory does not exist, it is created.

## Customization guide

- Changing the focal point and margin:  
  Edit `predict_sam2()` in sam_utilities.py. The central point is fixed; the two supporting points use a margin `m = int(min(w, h) * mask_threshold)`. To make the focal region tighter or wider, adjust `mask_threshold` in the call inside `process()` or change the point coordinates.

- Modifying the color vocabulary and prompts:  
  Update the `colors` list in main.py. Adjust `prefix_prompts` and `postfix_prompts` in `process()`. Because the color is extracted by searching for `f" {color} "` within the winning prompt, ensure your prefixes and postfixes include surrounding spaces so colors remain tokenized. For example, prefer “a dark ” instead of “a dark” to avoid “a darkred color”.

- Performance and accuracy trade-offs:  
  Reducing the prompt set (fewer prefixes/postfixes) speeds up CLIP scoring. Switching CLIP to a smaller model (e.g., “ViT-B/32”) will also increase throughput at potential cost to accuracy. GPU acceleration (CUDA or Apple MPS) is recommended.

- Paths and models:  
  To point to different SAM 2 models or configs, change `sam_model_name` and `sam_config_name` in main.py. The checkpoint and config paths are constructed by `get_checkpoint_path()` and `get_model_cfg_path()`.

## Practical tips
  
- For true transparent backgrounds, replace background painting with an alpha mask in `predict_sam2()`.

## Project structure

- main.py — entry point: device setup, model loading, processing loop  
- config.json - prefixes, items, postfixes
- clip_utilities.py — CLIP wrapper with `ClipModel.process()`  
- sam_utilities.py — SAM 2 helpers and `predict_sam2()` (focalization + background handling)  
- sam2/ — SAM 2 package (must include configs under `sam2/configs/sam2.1/`)  
- checkpoints/ — SAM 2 checkpoint (`sam2.1_hiera_large.pt`)  
- dataset_cars/ — input images  
- output/ — results organized by color
- **SAM 2** source code

## Using This Pipeline Beyond Cars

This pipeline is not limited to cars. The SAM 2 focalization step is class-agnostic and simply prioritizes the center of the frame; CLIP classifies color from text prompts you define. To repurpose the project for other object categories (e.g., apparel, footwear, furniture, food), adapt three things:

- Input folder: replace dataset_cars with your dataset directory.
- Prompt components: swap in domain-appropriate prefixes and postfixes, keeping the color list relevant to your objects.
- Color extraction rule: the code extracts the color by searching for " {color} " inside the winning prompt. Keep color tokens lowercase and ensure your prefix/postfix strings preserve a literal space on both sides of the color.

Technical guidelines:
- Prefixes should typically end with a trailing space, and postfixes should begin with a leading space so the composed prompt contains “ {color} ” (space-delimited). This ensures get_color() can find the color reliably.
- Avoid punctuation immediately adjacent to the color token (e.g., “ red,”) if you rely on the default get_color(). Consider making get_color() case-insensitive or tokenizing if you need more flexibility.
- Multi-word items (e.g., "space gray", "rose gold") work as long as you preserve surrounding spaces in the composed prompt.

Example domain presets

| Domain | Example items (lowercase) | Example prefixes (end with space) | Example postfixes (start with space) |
|---|---|---|---|
| Apparel (shirts/tops) | navy, beige, olive, burgundy, charcoal, cream, khaki, teal | "a ", "a shirt in ", "a t-shirt that is ", "a garment colored in " | " shirt", " t-shirt", " top", " garment" |
| Footwear | black, white, brown, tan, navy, gray | "a ", "a pair of ", "a shoe in ", "a sneaker that is " | " shoe", " sneaker", " boot", " footwear" |
| Furniture | black, white, walnut, oak, mahogany, gray | "a ", "a chair in ", "a sofa that is ", "a table finished in " | " chair", " sofa", " table", " furniture piece" |
| Consumer electronics | black, silver, space gray, rose gold, midnight, starlight | "a ", "a device in ", "a phone that is ", "a laptop finished in " | " device", " phone", " laptop", " product" |
| Fruits/produce | red, green, yellow, golden, purple, orange | "a ", "a ripe ", "a fresh ", "a photo of a " | " apple", " banana", " mango", " grape" |

Concrete, ready-to-use lists

- Apparel
  - Items:
    - ["navy", "beige", "olive", "burgundy", "charcoal", "cream", "khaki", "teal"]
  - Prefixes (note the trailing spaces):
    - ["a ", "a shirt in ", "a t-shirt that is ", "a garment colored in "]
  - Postfixes (note the leading spaces):
    - [" shirt", " t-shirt", " top", " garment"]

- Footwear
  - Items:
    - ["sports ", "runner's", "olympics", "casual"]
  - Prefixes:
    - ["a ", "a pair of ", "a shoe in ", "a sneaker that is "]
  - Postfixes:
    - [" shoe", " sneaker", " boot", " footwear"]

- Electronics
  - Colors:
    - ["iPhone", "Samsung", "Motorola", "Xiaomi"]
  - Prefixes:
    - ["a ", "a device in ", "a phone that is ", "a laptop finished in "]
  - Postfixes:
    - [" device", " phone", " laptop", " product"]

## Contact and attribution

- Author: https://github.com/iamdroppy  
- Project page: https://github.com/iamdroppy/sam2vit  
- SAM 2: https://github.com/facebookresearch/sam2/  
- CLIP: https://github.com/openai/CLIP  
- License: [PLACEHOLDER] (add your project’s license)
