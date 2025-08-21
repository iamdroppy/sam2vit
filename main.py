import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import json
import time
import argparse
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image
from rich.table import Table as RichTable
from rich import console as rich_console
from rich import print as rprint

from cuda_device import device_check
from loguru import logger
from clip_model import ClipModel
from image_processor import process_sam2, process_image, process_yolo
from sam_model import SamModel

from config import Config, load_config

from utils import get_args, elapsed_time_to_string

_LOG_LEVEL_MAP = {"TRACE": 0, "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

table = RichTable(title="SAM2 with CLIP Results", show_lines=True, show_header=True)
table.add_column("Status", style="")
table.add_column("Item", style="bold")
table.add_column("Prompt", style="black bold")
table.add_column("Probability", style="green")
table.add_column("File Name", style="dim")

def main(args: argparse.Namespace):
    """
    Initializes the main function for the script.
    This function sets up the environment, checks the device, loads models, and processes images.
    Passing the `--debug` or `-d` argument enables debug mode, which provides additional logging information.
    """
    config = load_config("config.json")
    if not config:
        logger.critical("Failed to load configuration from config.json. Exiting.")
        exit(ExitCodes.MISSING_CONFIG)
    device_clip = torch.device(args.device)
    device = torch.device(args.device)

    sam_model_name = args.sam_model if args.sam_model else "sam2.1_hiera_large"
    sam_config_name = args.sam_config if args.sam_config else "sam2.1_hiera_l"
    clip_model_name = args.clip_model if args.clip_model else "ViT-L/14@336px"
    seed = args.seed if args.seed else 3

    device_check(device, device_clip)
    logger.info(f"Using device: `{device.type}` for ✏️ SAM2 and `{device_clip.type}` for CLIP")

    logger.info(f"🌱 Seed: {args.seed}")
    np.random.seed(seed)

    # Load Models
    logger.info(f"Loading CLIP model: {clip_model_name} on device: {device_clip.type}")
    if device.type == "cuda" and clip_model_name == "ViT-L/14@336px":
        logger.warning("Using default CLIP model ViT-L/14@336px, which is heavy on memory. Consider using a smaller model for better performance.")
    elif device.type == "cpu" and (clip_model_name == "ViT-B/32" or clip_model_name == "ViT-L/14@336px"):
        logger.critical(f"Using default CLIP model {clip_model_name}, which is VERY heavy on memory and on CPU. Consider using a smaller model for better performance for CPU or consider a CUDA GPU.")
    
    clip_model = ClipModel(clip_model_name, device_clip)
    sam2_model = None
    if args.no_sam:
        logger.warning(f"Skipping SAM2 model: {sam_model_name} with config: {sam_config_name} on device: {device.type}")
    else:
        if device.type == "cuda" and sam_model_name == "sam2.1_hiera_large":
            logger.warning("Using default SAM2 model sam2.1_hiera_large, which is heavy on memory. Consider using a smaller model for better performance.")
        elif device.type == "cpu" and sam_model_name == "sam2.1_hiera_large":
            logger.critical("Using default SAM2 model sam2.1_hiera_large, which is VERY heavy on memory and on CPU. Consider using a smaller model for better performance for CPU or consider a CUDA GPU.")

        logger.info(f"Loading SAM2 model: {sam_model_name} with config: {sam_config_name} on device: {device.type}")
        sam2_model = SamModel(device=device, config=config, config_name=sam_config_name, checkpoint_path=sam_model_name)

    return process(args, clip_model, sam2_model, config)



def process(args: argparse.Namespace, clip_model: ClipModel, sam2_model: SamModel, config: Config) -> Dict[str, int]:
    """
    Processes images in the '--dataset-dir' directory using the SAM2 model for segmentation and CLIP model for classification.
    Outputs segmented images to the '--output-dir' directory, organized by identified item categories using the CLIP and SAM2 models.
    Args:
        clip_model (ClipModel): 
            An instance of the ClipModel class used for processing images.
        sam2_model: The SAM2 model used for image segmentation.
    """
    # Load configuration from config.json
    prefixes = config.prefixes
    items = config.items
    postfixes = config.postfixes

    prompt_list = config.get_prompts()
    prompts = [prompt for prompt in prompt_list]
    logger.debug(f"Generated {len(prompts)} prompts from config.")
    logger.trace(f"10 Random Sample prompts: {', '.join(np.random.choice(prompts, 10))}")
    logger.info(f"Processing images in {args.input_dir} 📂 folder...")
    
    i = 0
    errors = 0
    success = 0
    files = os.listdir(args.input_dir)
    if not files or files == []:
        logger.critical(f"No files found in directory: {args.input_dir}")
        exit(ExitCodes.NO_INPUT_FILES)
    wrong_ext_advised = False
    for file in files:
        try:
            i += 1
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                if not wrong_ext_advised:
                    logger.warning(f"Skipping non-image file: `{file}`: Only .png, .jpg, and .jpeg files are processed. This message will only be shown once.")
                    wrong_ext_advised = True
                continue
            
            logger.trace(f"({success} OK/{errors} NOK) Processing image: {file}")
            
            img_path = os.path.join(args.input_dir, file)
            img = Image.open(img_path)
            
            # process SAM2 + CLIP
            result_sam = process_sam2(args, img, sam2_model)
            result = process_image(result_sam, config, prompts, clip_model)

            if result is not None:
                if result["item"] in items:
                    os.makedirs(os.path.join(args.output_dir, result["item"]), exist_ok=True)
                    output_path = os.path.join(args.output_dir, result["item"], file.replace(".jpg", ".png").replace(".jpeg", ".png"))
                    out: Image = result["segmented_image"]
                    if args.output_original:
                        out: Image = img
                    out.save(output_path)
                    logger.debug(f"{result['item'].capitalize() if result['item'] else 'Unknown'} ({result['probability']:4f}%)\n\t\t\t\t\t\t--> `{result['prompt']}` ")
                    success += 1
                else:
                    logger.warning(f"Can't find prompt in {file} with prompt: {result['prompt']}")
                    errors += 1
                
            else:
                logger.warning(f"Failed to process image: {file}")
        except Exception as e:
            logger.error(e)
            errors += 1
    if success == 0 and errors >= 0:
        logger.critical(f"No images were processed (with {errors}). Please check the input directory.")
    elif success == 0 and errors >= 0:
        logger.error(f"There are {errors} in {i} files (no success).")
        
    logger.trace(f"Processed {i} files with {success} succeeded and {errors} errors.")
    if success > errors:
        ratio = success / errors
        logger.success(f"Finished process: Success/Error ratio: {ratio:.2f} in {i} files")
    elif errors > success:
        ratio = errors / success
        logger.warning(f"Finished process: Error/Success ratio: {ratio:.2f} in {i} files")
    return {
        "processed_files": i,
        "errors": errors,
        "processed_images": i - errors
    }
    

class ExitCodes:
    SUCCESS = 0
    INVALID_LOG_LEVEL = -13
    NO_INPUT_FILES = -41,
    MISSING_CONFIG = -55

if __name__ == "__main__":
    args = get_args()
    logger.remove()
    rprint(f"Starting [bold green]sam[/bold green][bold blue]2[/bold blue][bold green]vit[/bold green]")
    rprint(f"[green]+[/green] [bold green]positive scale pin:[/bold green] {args.positive_scale_pin}")
    rprint(f"[red]+[/red] [bold red]negative scale pin:[/bold red] {args.negative_scale_pin}")
    rprint("[green]-[/green]-" * 50)
    logger.add(sys.stderr, level=args.log_level.upper())
    name = args.file_log_name if args.file_log_name else "app.log"
    if args.file_log_no_reset:
        os.remove(name)
        logger.trace(f"The log file '{name}' was reset on boot.")
    logger.add(name, level=args.file_log_level.upper(), rotation=args.file_log_rotation if args.file_log_rotation else "100 MB", enqueue=True)
    level = 0
    if args.log_level.upper() not in _LOG_LEVEL_MAP:
        logger.critical("Invalid log level specified. Use one of: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL.")
        exit(ExitCodes.INVALID_LOG_LEVEL)

    try:
        import cv2
        from matplotlib import pyplot as plt
    except ImportError:
        plt = None
        cv2 = None
        if args.show_image:
            logger.warning("matplotlib and cv2 are required for visualization but are not installed.")
            args.show_image = False

    if args.device and args.device not in ["cuda", "cpu", "mps"]:
        raise ValueError("Invalid device specified. Use 'cuda', 'cpu', or 'mps'.")
    
    if args.device == "mps":
        logger.warning(f"MPS device is not fully tested.")

    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}. Please provide a valid directory.")
        exit(ExitCodes.NO_INPUT_DIR)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()
    result = {
        "processed_files": 0,
        "errors": 0,
        "processed_images": 0
    }
    
    main(args)
    
    rich_console.Console().print(table)
    
    end = time.time()
    elapsed_time = end - start
    
    logger.success(f"Total processing time: {elapsed_time_to_string(elapsed_time)}")
    images_per_second = result["processed_images"] / elapsed_time if elapsed_time > 0 else 0
    logger.success(f"Images per second: {images_per_second:.2f} images/sec")
    logger.success(f"Images processed: {result['processed_images']}")
    logger.success(f"Output directory: {args.output_dir}")