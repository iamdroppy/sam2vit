import argparse
from typing import List, Dict, Any, Optional
from config import Config
from clip_model import ClipModel
from sam_model import SamModel
from loguru import logger
import os
from numpy import ndarray
from PIL import Image

def process_image(args: argparse.Namespace,
                  file: str,
                  config: Config,
                  prompts: List[str],
                  clip_model: ClipModel,
                  sam2_model: SamModel):
    try:
        img_path = os.path.join(args.input_dir, file)
        img = Image.open(img_path)
        if not args.no_sam:
            segmented_image = sam2_model.predict_sam2(
                img, mask_threshold=args.negative_scale_pin, show=False, scale_pin=args.positive_scale_pin
            )
            if segmented_image is None:
                return None
        else:
            segmented_image = img
            
        clip_result = clip_model.process(segmented_image, prompts)
        prompt = clip_result['prompt']
        probability = clip_result['probability_percentage']
        item = config.get_item_from_list(prompt)

        return {
            "segmented_image": segmented_image,
            "file": file,
            "prompt": prompt,
            "probability": probability,
            "item": item,
            "clip_result": clip_result
        }
    except Exception as e:
        raise RuntimeError(f"Failed to process image {file}: {e}") from e
    return None

def process_yolo(args: argparse.Namespace,
                 file: str,
                 config: Config,
                 prompts: List[str],
                 clip_model: ClipModel,
                 sam2_model: SamModel,
                 result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return None