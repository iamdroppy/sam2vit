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
from rich import print as rich_print
from rich.table import Table as RichTable
from rich import console as rich_console

from cuda_device import device_check
from loguru import logger
from sam_model import predict_sam2, get_checkpoint_path, get_model_cfg_path
from clip_model import ClipModel
from sam_model import SamModel
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import Config, load_config

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
    device_clip = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    sam_model_name = args.sam_model if args.sam_model else "sam2.1_hiera_large"
    sam_config_name = args.sam_config if args.sam_config else "sam2.1_hiera_l"
    clip_model_name = args.clip_model if args.clip_model else "ViT-L/14@336px"
    seed = args.seed if args.seed else 3

    device_check(device, device_clip)
    logger.info(f"Using device: `{device.type}` for ✏️ SAM2 and `{device_clip.type}` for CLIP")

    sam2_checkpoint = get_checkpoint_path(sam_model_name)
    model_cfg = get_model_cfg_path(sam_config_name)

    logger.success(f"🌱 Seed: {args.seed}")
    np.random.seed(seed)

    # Load Models
    logger.info(f"Loading CLIP model: {clip_model_name} on device: {device_clip.type}")
    if clip_model_name == "ViT-L/14@336px":
        logger.warning("Using default CLIP model ViT-L/14@336px, which is heavy on memory. Consider using a smaller model for better performance.")
    clip_model = ClipModel(clip_model_name, device_clip)
    sam2_model = None
    if args.no_sam:
        logger.warning(f"Skipping SAM2 model: {sam_model_name} with config: {sam_config_name} on device: {device.type}")
    else:
        logger.info(f"Loading SAM2 model: {sam_model_name} with config: {sam_config_name} on device: {device.type}")
        sam2_model = SamModel(build_sam2(model_cfg, sam2_checkpoint, device=device))

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
    global _prefixes, _items, _postfixes
    
    _prefixes = config.prefixes
    _items = config.items
    _postfixes = config.postfixes

    prompt_list = config.get_prompts()
    prompts = [prompt for prompt in prompt_list]
    logger.debug(f"Generated {len(prompts)} prompts from config.")
    logger.trace(f"10 Random Sample prompts: {', '.join(np.random.choice(prompts, 10))}")
    logger.info(f"Processing images in {args.dataset_dir} 📂 folder...")
    
    i = 0
    errors = 0
    success = 0
    files = os.listdir(args.dataset_dir)
    if not files or files == []:
        logger.critical(f"No files found in directory: {args.dataset_dir}")
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
            logger.debug(f"Processing image: {file}")
            img_path = os.path.join(args.dataset_dir, file)
            img = Image.open(img_path)
            if not args.no_sam:
                segmented_image = predict_sam2(
                    img, sam2_model, mask_threshold=0.05, show=False
                )
                if segmented_image is None:
                    logger.error(f"Failed to segmentate image: {file}")
                    continue
                logger.debug(f"Processed image: {file}")
            else:
                logger.debug(f"Skipping SAM2 segmentation for image: {file}")
                segmented_image = img
                

            clip_result = clip_model.process(segmented_image, prompts)
            prompt = clip_result['prompt']
            probability = clip_result['probability_percentage']
            item = config.get_item_from_list(prompt)

            logger.debug(f"{item.capitalize() if item else 'Unknown'} `{prompt}` {probability:4f}%")

            if item in _items:
                if (config.yolo.enabled):
                    os.makedirs(os.path.join(args.output_dir, item), exist_ok=True)
                    output_path = os.path.join(args.output_dir, item, file.replace(".jpg", ".png"))
                    segmented_image.save(output_path)
                    success += 1
            else:
                logger.warning(f"Can't find prompt in {file} with prompt: {prompt}")
                errors += 1
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
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
    
    default_sam_model_name = "sam2.1_hiera_large"
    default_sam_config_name = "sam2.1_hiera_l"
    default_clip_model_name = "ViT-L/14@336px"
    default_seed = 3
    
    arg_parsed = argparse.ArgumentParser(description="Run SAM2 with CLIP for image segmentation and classification.")
    
    # defaults
    # SAM2
    arg_parsed.add_argument("--no-sam", "-x", default=False, action="store_true", help="Disables the Segmentation")
    arg_parsed.add_argument("--sam-model", default=default_sam_model_name, type=str, help="SAM2 model name (default: sam2.1_hiera_large)")
    arg_parsed.add_argument("--sam-config", default=default_sam_config_name, type=str, help="SAM2 config name (default: sam2.1_hiera_l)")
    # CLIP
    arg_parsed.add_argument("--clip-model", "-c", default=default_clip_model_name, type=str, help="CLIP model name (default: ViT-L/14@336px)")
    
    arg_parsed.add_argument("--use-yolo-model", "-y", default=False, action="store_true", help="Use YOLO model for object detection")
    # np
    arg_parsed.add_argument("--seed", "-S", default=default_seed, type=int, help="Random seed for reproducibility (default: 3)")
    arg_parsed.add_argument("--show-image", "-g", action="store_true", default=False, help="Show the output image after each processing step")
    
    # io
    arg_parsed.add_argument("--input-dir", "-i", default="_input", type=str, required=True, help="Path to the dataset directory (default: _input)")
    arg_parsed.add_argument("--output-dir", "-o", default="_output", type=str, required=True, help="Path to the output directory (default: _output)")

    # logs
    arg_parsed.add_argument("--log-level", "-l", type=str, default="INFO", choices=["WORK", "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Sets the log level")
    arg_parsed.add_argument("--file-log-level", "-p", type=str, default="TRACE", choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Sets the log level of `app.log` file (default: trace)")
    arg_parsed.add_argument("--file-log-name", "-n", type=str, default="app.log", help="Sets the log file name (default: app.log)")
    arg_parsed.add_argument("--file-log-rotation", "-r", type=str, default="100 MB", help="Sets the log file rotation size (default: 100 MB)")
    arg_parsed.add_argument("--seed", "-s", default=3, type=int, help="Random seed for reproducibility (default: 3)")
    args = arg_parsed.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())
    logger.add(args.file_log_name if args.file_log_name else "app.log", level=args.file_log_level.upper(), rotation=args.file_log_rotation if args.file_log_rotation else "50 MB", enqueue=True)
    logger.level("MODEL", no=4, color="<red>", icon="!!!")
    logger.log("MODEL", "A user updated some information.")
    level = 0
    if args.log_level.upper() not in _LOG_LEVEL_MAP:
        logger.critical("Invalid log level specified. Use one of: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL.")
        exit(ExitCodes.INVALID_LOG_LEVEL)
        
    if args.log_level.upper() == "WORK":
        logger.log(4, "Logging in WORK mode.")

    try:
        import cv2
        from matplotlib import pyplot as plt
    except ImportError:
        plt = None
        cv2 = None
        if args.show_image:
            logger.warning("matplotlib and cv2 are required for visualization but are not installed.")
            args.show_image = False

    if args.force_device and args.force_device not in ["cuda", "cpu", "mps"]:
        raise ValueError("Invalid device specified. Use 'cuda', 'cpu', or 'mps'.")
    
    if args.device == "mps":
        logger.warning(f"MPS device is not fully tested.")

    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}. Please provide a valid directory.")
        exit(ExitCodes.NO_INPUT_DIR)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting the sam2vit...")
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
    # if it reaches 1 ms, show 1ms, also if it reaches 1 second, also show 1.00 seconds, if it also reaches 1 minute, show 1.00 minutes
    def elapsed_time_to_string(elapsed_time):
        """
        Converts elapsed time in seconds to a human-readable string format.
        Handles milliseconds, seconds, minutes, and hours.
        """
        elapsed_parts = []
        if elapsed_time < 1:
            elapsed_parts.append(f"{elapsed_time * 1000:.2f} ms")
            elapsed_time = elapsed_time * 1000  # convert to milliseconds
        if elapsed_time >= 1:
            elapsed_parts.append(f"{elapsed_time:.2f} seconds")
            elapsed_time = elapsed_time - int(elapsed_time)
        if elapsed_time >= 60:
            elapsed_parts.append(f"{elapsed_time:.2f} minutes")
            elapsed_time = elapsed_time - int(elapsed_time)
        if elapsed_time >= 3600:
            elapsed_parts.append(f"{elapsed_time / 3600:.2f} hours")
            elapsed_time = elapsed_time - int(elapsed_time)

        return ", ".join(elapsed_parts)
    
    logger.success(f"Total processing time: {elapsed_time_to_string(elapsed_time)}")
    images_per_second = result["processed_images"] / elapsed_time if elapsed_time > 0 else 0
    logger.success(f"Images per second: {images_per_second:.2f} images/sec")
    logger.success(f"Images OK: {result['processed_images']}")
    logger.success(f"Output directory: {args.output_dir}")