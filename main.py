import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json
import time
import argparse
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image
from rich import print
from rich.table import Table as RichTable
from rich import console as rich_console

from cuda_device import device_check
from console import Console
from sam_utilities import predict_sam2, get_checkpoint_path, get_model_cfg_path
from clip_utilities import ClipModel
from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor  # unused import

def get_prompts() -> List[str]:
    """
    Generate a list of prompt strings by combining each item with every prefix and postfix.

    Returns:
        List[str]: All combinations of prefix + item + postfix.
    """
    prompts: List[str] = []
    for item in _items:
        for prefix in _prefixes:
            for postfix in _postfixes:
                prompts.append(f"{prefix.strip()} {item.strip()} {postfix.strip()}")
                
    with open(".prompts.log", "w") as f:
        f.write("\n".join(prompts))
        
    return prompts
 
_items: List[str] = []
_prefixes: List[str] = []
_postfixes: List[str] = []

def get_item_from_list(prompt: str) -> Optional[str]:
    """
    Extracts the first item from a prompt string.
    Args:
        prompt (str): 
            A string that may contain an item name.
    """
    for item in _items:
        if " " + item + " " in prompt:
            return item
    return None

table = RichTable(title="SAM2 with CLIP Results", show_lines=True, show_header=True)
table.add_column("Status", style="")
table.add_column("Item", style="bold")
table.add_column("Prompt", style="black bold")
table.add_column("Probability", style="green")
table.add_column("File Name", style="dim")

def main(args: argparse.Namespace) -> Dict[str, int]:
    """
    Initializes the main function for the script.
    This function sets up the environment, checks the device, loads models, and processes images.
    Passing the `--debug` or `-d` argument enables debug mode, which provides additional logging information.
    """
    if "--debug" in os.sys.argv or "-d" in os.sys.argv:
        Console.is_debug = True
        Console.debug("Debug mode is enabled.")
    else:
        Console.info("Debug mode is off. Use --debug or -d to enable it.")

    device_clip = torch.device("cuda" if torch.cuda.is_available() or args.force_device == "cuda" else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() or args.force_device == "mps" else "cuda" if torch.cuda.is_available() or args.force_device == "cuda" else "cpu")
    
    if (args.force_device):
        Console.warning(f"Using forced device: [red]{args.force_device}[/red]")

    sam_model_name = args.sam_model if args.sam_model else "sam2.1_hiera_large"
    sam_config_name = args.sam_config if args.sam_config else "sam2.1_hiera_l"
    clip_model_name = args.clip_model if args.clip_model else "ViT-L/14@336px"
    seed = args.seed if args.seed else 3

    device_check(device, device_clip)
    Console.info(f"[red][bold]Using device:[/bold][/red] [bold]{device.type}[/bold] for ✏️ SAM2 and [bold]{device_clip.type}[/bold] for 🔗 CLIP")

    sam2_checkpoint = get_checkpoint_path(sam_model_name)
    model_cfg = get_model_cfg_path(sam_config_name)

    Console.success(f"🌱 [green]Setting seed:[/green] [bold green]{seed}[/bold green]")
    np.random.seed(seed)

    # Load Models
    Console.clip(f"🔃 [grey]Loading [bold]CLIP[/bold] model:[/grey] [bold]{clip_model_name}[/bold] on device: [bold]{device_clip.type}[/bold]")
    clip_model = ClipModel(clip_model_name, device_clip)
    sam2_model = None
    if args.no_sam:
        Console.warning(f"🔃 [yellow]Skipping [/yellow][red]SAM2[/red] [yellow]model:[/yellow] [bold]{sam_model_name}[/bold]")
    else:
        Console.sam2(f"🔃 Loading SAM2 model: [bold]{sam_model_name}[/bold] with config: [bold]{sam_config_name}[/bold] on device: [bold rosy_brown]{device.type}[/bold][/rosy_brown]")
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    return process(args, clip_model, sam2_model)

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Loads the configuration from a JSON file.
    Args:
        file_path (str): The path to the JSON configuration file.
    Returns:
    Dict[str, Any]: The loaded configuration as a dictionary.
    """
    with open(file_path, "r") as f:
        return json.load(f)

def process(args: argparse.Namespace, clip_model: ClipModel, sam2_model) -> Dict[str, int]:
    """
    Processes images in the '--dataset-dir' directory using the SAM2 model for segmentation and CLIP model for classification.
    Outputs segmented images to the '--output-dir' directory, organized by identified item categories using the CLIP and SAM2 models.
    Args:
        clip_model (ClipModel): 
            An instance of the ClipModel class used for processing images.
        sam2_model: The SAM2 model used for image segmentation.
    """
    # Load configuration from config.json
    config = load_config("config.json")
    global _prefixes, _items, _postfixes
    _prefixes = config.get("prefixes", ["an"])
    _items = config.get("items", ["object"])
    _postfixes = config.get("postfixes", ["there"])

    prompt_list = get_prompts()
    prompts = prompt_list  # no need to copy
    Console.debug(f"⭕ [rosy_brown]Generated prompts: [bold]{len(prompts)}[/bold][rosy_brown] prompts[/rosy_brown]")
    Console.debug(f"🔍 [rosy_brown]2 Sample prompts: [bold]{', '.join(prompts[:2])}[/bold][rosy_brown]...[/rosy_brown]")
    Console.debug(f"🔃 [rosy_brown]Processing images[/rosy_brown] in [rosy_brown][bold]{args.dataset_dir}[/bold][/rosy_brown] 📂 [yellow]folder[/yellow]...")

    i = 0
    errors = 0
    for file in os.listdir(args.dataset_dir):
        try:
            i += 1
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                Console.debug(f"[red]Skipping [bold]non-image[/bold] file[/red]: [bold]{args.dataset_dir}/{file}[/bold] - [bold underline]Only .png, .jpg, and .jpeg files are processed.[/bold underline]")
                continue
            Console.debug(f"🔄 Processing file: [bold]{file}[/bold]")
            img_path = os.path.join(args.dataset_dir, file)
            img = Image.open(img_path)
            if not args.no_sam:
                segmented_image = predict_sam2(
                    img, sam2_model, mask_threshold=0.05, show=False
                )
                if segmented_image is None:
                    Console.error(f"Failed to segmentate image: [bold]{file}[/bold]")
                    continue
                Console.debug(f"Processed image: [bold]{file}[/bold]")
            else:
                Console.debug(f"⚠️ [yellow]Skipping SAM2 segmentation for image: [bold]{file}[/bold][/yellow]")
                segmented_image = img
                

            clip_result = clip_model.process(segmented_image, prompts)
            prompt = clip_result['prompt']
            probability = clip_result['probability_percentage']
            item = get_item_from_list(prompt)

            Console.debug(
                f"[bold underline]{item.capitalize() if item else 'Unknown'}[/bold underline] "
                f"[light_green]`[/light_green][grey]{prompt}[/grey][light_green]`[/light_green] "
                f"[orange3]{probability:.2f}[/orange3][light_pink3]%[/light_pink3]"
            )

            if item in _items:
                os.makedirs(os.path.join(args.output_dir, item), exist_ok=True)
                output_path = os.path.join(args.output_dir, item, file.replace(".jpg", ".png"))
                segmented_image.save(output_path)
                Console.debug(f"Saved image to: [bold]{output_path}[/bold]")
                Console.success(
                    f"🗃️ [bold]{file}[/bold] [chartreuse2]{item}[/chartreuse2] from [bold]{prompt}[/bold]"
                )
            else:
                Console.warning(
                    f"Can't find prompt in [bold]{file}[/bold] with prompt: [bold]{prompt}[/bold]"
                )
                errors += 1
        except Exception as e:
            Console.error(f"Error processing file [bold]{file}[/bold]: {e}")
            errors += 1
    Console.debug(f"🔚 Processed {i} files with {errors} errors.")
    return {
        "processed_files": i,
        "errors": errors,
        "processed_images": i - errors
    }

if __name__ == "__main__":
    
    default_sam_model_name = "sam2.1_hiera_large"
    default_sam_config_name = "sam2.1_hiera_l"
    default_clip_model_name = "ViT-L/14@336px"
    default_seed = 3
    
    arg_parsed = argparse.ArgumentParser(description="Run SAM2 with CLIP for image segmentation and classification.")
    arg_parsed.add_argument("--sam-model", "-s", default=default_sam_model_name, type=str, help="SAM2 model name (default: sam2.1_hiera_large)")
    arg_parsed.add_argument("--sam-config", "-c", default=default_sam_config_name, type=str, help="SAM2 config name (default: sam2.1_hiera_l)")
    arg_parsed.add_argument("--no-sam", "-x", default=False, action="store_true", help="Disables the Segmentation")
    arg_parsed.add_argument("--clip-model", "-m", default=default_clip_model_name, type=str, help="CLIP model name (default: ViT-L/14@336px)")
    arg_parsed.add_argument("--seed", "-S", default=default_seed, type=int, help="Random seed for reproducibility (default: 3)")
    arg_parsed.add_argument("--dataset-dir", "-i", required=True, type=str, help="Path to the dataset directory (default: dataset_cars)")
    arg_parsed.add_argument("--output-dir", "-o", default="output", type=str, help="Path to the output directory (default: output)")
    arg_parsed.add_argument("--no-log", "-n", action="store_true", default=False,help="Disable logging to main.log")
    arg_parsed.add_argument("--debug", '-d', action="store_true", default=False, help="Enable debug mode for additional logging")
    arg_parsed.add_argument("--verbose", "-v", action="store_true", default=False, help="Enable verbose mode (sets debug mode)")
    arg_parsed.add_argument("--show-image", "-g", action="store_true", default=False, help="Show the output image after each processing step")
    arg_parsed.add_argument("--force-device", "-f", default=False, type=str, help="Force the use of a specific device (cuda, cpu)")
    arg_parsed.add_argument("--log", '-l', type=str, default="sam2vit.log", help="Log file path (default: sam2vit.log)")
    args = arg_parsed.parse_args()
    
    if args.force_device and args.force_device not in ["cuda", "cpu", "mps"]:
        raise ValueError("Invalid device specified. Use 'cuda', 'cpu', or 'mps'.")
    
    if args.force_device:
        Console.warning(f"Using forced device: [red]{args.force_device}[/red]")
    
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    
    if (args.verbose and not args.debug):
        args.debug = True
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔔 Starting the sam2vit...")
    start = time.time()
    result = {
        "processed_files": 0,
        "errors": 0,
        "processed_images": 0
    }
    stream = None
    if args.no_log:
        Console.warning("Logging is disabled. Use [yellow]--[/yellow][yellow dim]no-log[/yellow][/dim] to enable logging.")
        result = main(args)
    else:
        stream = open(args.log, "w", encoding="utf-8")
        stream.write(f"\n\nLogging started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        Console.set_stream(stream)
        result = main(args)
        Console.info(f"Logging completed. Check {args.log} for details.")
        stream.flush()
            
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
    import sys
    Console.success(f"⏱️ [bold green]Total processing time:[/bold green] {elapsed_time_to_string(elapsed_time)}")
    images_per_second = result["processed_images"] / elapsed_time if elapsed_time > 0 else 0
    Console.success(f"📸 [bold green]Images per second:[/bold green] {images_per_second:.2f} images/sec")
    Console.success(f"📂 [bold green]Images OK:[/bold green] {result['processed_images']}")
    Console.success(f"📂 [bold green]Output directory:[/bold green] {args.output_dir}")
    # close stream if not None
    if stream and not stream.closed:
        stream.close()