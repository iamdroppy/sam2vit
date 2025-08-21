import argparse
from typing import List, Dict, Any, Optional
from config import Config
from clip_model import ClipModel
from sam_model import SamModel
from loguru import logger
import os
from numpy import ndarray
from PIL import Image

def process_sam2(args: argparse.Namespace,
                 img: Image.Image,
                 sam2_model: SamModel) -> Image:
    segmented_image = None
    if not args.no_sam:
        segmented_image = sam2_model.predict_sam2(
            img, mask_threshold=args.negative_scale_pin, show=False, scale_pin=args.positive_scale_pin
        )
    return segmented_image if segmented_image is not None else img


def process_image(img: Image.Image,
                  config: Config,
                  prompts: List[str],
                  clip_model: ClipModel):
    try:
        clip_result = clip_model.process(img, prompts)
        prompt = clip_result['prompt']
        probability = clip_result['probability_percentage']
        item = config.get_item_from_list(prompt)

        return {
            "segmented_image": img,
            "prompt": prompt,
            "probability": probability,
            "item": item,
            "clip_result": clip_result
        }
    except Exception as e:
        raise RuntimeError(f"Failed to process image CLIP: {e}") from e

def process_yolo(args: argparse.Namespace,
                 file: str,
                 config: Config,
                 prompts: List[str],
                 clip_model: ClipModel,
                 sam2_model: SamModel,
                 result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    
    return None

if __name__ == "__main__":
    pass