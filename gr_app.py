import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image
import gradio as gr

from clip_model import ClipModel
from image_processor import process_sam2, process_image
from sam_model import SamModel
from yolo_model import YoloModel


class Definitions:
    clip_model: ClipModel = None
    sam2_model: SamModel = None
    yolo_model: YoloModel = None

    @staticmethod
    def setup(
        sam_model_name: str,
        sam_config_name: str,
        clip_model_name: str,
        yolo_model_name: str,
        prefixes: List[str],
        items: List[str],
        postfixes: List[str],
        yolo_prompts: List[str] = [],
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        Definitions.clip_model = ClipModel(clip_model_name, device=device)

        cfg = {
            "csv_prefixes": [],
            "csv_postfixes": [],
            "prefixes": prefixes,
            "items": items,
            "postfixes": postfixes,
            "yolo_model": yolo_model_name or "yolo11x-seg.pt",
            "yolo_confidence_threshold": 0.5,
            "yolo_prompts": yolo_prompts,
        }
        Definitions.sam2_model = SamModel(
            device=device,
            config_name=sam_config_name,
            checkpoint_path=sam_model_name,
            cfg_dict=cfg,
        )
        Definitions.yolo_model = YoloModel(Definitions.sam2_model.config)

    @staticmethod
    def update_cfg(cfg_dict) -> None:
        if Definitions.sam2_model is None:
            return
        Definitions.sam2_model.update_cfg(cfg_dict)


def normalize_list(value) -> List[str]:
    """
    Normalize input from gr.List (which can be None, list[str], list[list], etc.)
    to a clean list[str] without empty strings.
    """
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for v in value:
            if isinstance(v, list):
                if len(v) == 0:
                    continue
                s = str(v[0]).strip()
            else:
                s = str(v).strip()
            if s:
                result.append(s)
        return result
    s = str(value).strip()
    return [s] if s else []


def add_list_row(lst):
    lst = lst or []
    if len(lst) > 0 and isinstance(lst[0], list):
        return lst + [[""]]
    return lst + [""]


def run_segmentation_and_classification(
    args,  # argparse.Namespace via gr.State
    image: Image.Image,
    prefixes_in,
    items_in,
    postfixes_in,
    yolo_prompts_in,
    seed: int,
) -> Tuple[Optional[Image.Image], str]:
    try:
        if Definitions.clip_model is None or Definitions.sam2_model is None:
            return None, "Models are not initialized. Please restart or check setup."

        if image is None:
            return None, "Please provide an input image."

        prefixes = normalize_list(prefixes_in)
        items = normalize_list(items_in)
        postfixes = normalize_list(postfixes_in)
        yolo_prompts = normalize_list(yolo_prompts_in)

        if len(items) == 0:
            return None, "Please add at least one Item."
        if len(prefixes) == 0:
            return None, "Please add at least one Prefix."
        if len(postfixes) == 0:
            return None, "Please add at least one Postfix."

        cfg = {
            "csv_prefixes": [],
            "csv_postfixes": [],
            "prefixes": prefixes,
            "items": items,
            "postfixes": postfixes,
            "yolo_model": getattr(args, "yolo_model", "yolo11x-seg.pt"),
            "yolo_confidence_threshold": 0.5,
            "yolo_prompts": yolo_prompts,
        }
        Definitions.update_cfg(cfg)

        try:
            np.random.seed(int(seed))
        except Exception:
            np.random.seed(0)

        result_seg = image
        if not getattr(args, "no_sam", False) and Definitions.sam2_model is not None:
            result_seg = process_sam2(args, image, Definitions.sam2_model)
            if result_seg is None:
                return None, "Segmentation failed."

        if getattr(args, "post_process_yolo", False) and Definitions.yolo_model is not None:
            yolo_img, label, confidence = Definitions.yolo_model.predict(result_seg)
            if yolo_img is not None:
                result_seg = yolo_img
            else:
                return None, "YOLO post-processing required but no valid detection was made."

        prompt_list = Definitions.sam2_model.config.get_prompts()
        prompts = [p for p in prompt_list]

        result = process_image(result_seg, Definitions.sam2_model.config, prompts, Definitions.clip_model)
        if result is None:
            return None, "No valid segmentation or classification result."

        predicted_item = result.get("item")
        if items and predicted_item not in items:
            return None, f"Predicted item '{predicted_item}' is not in the allowed items list."

        segmented_image = result.get("segmented_image")
        probability = result.get("probability", 0.0)
        prompt_used = result.get("prompt", "")

        text = f"Prompt: {prompt_used}\nProbability: {probability:.2f}%\nItem: {predicted_item}"
        return segmented_image, text

    except Exception as e:
        logger.exception("Error during segmentation/classification.")
        return None, f"Error: {e}"


def create_interface(args: argparse.Namespace) -> gr.Blocks:
    with gr.Blocks(title="SAM2 + CLIP + YOLO Segmentation and Classification") as ui:
        args_state = gr.State(args)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")

                item_table = gr.List(label="Item CLIP", headers=["Item"], col_count=1, value=[])
                item_add_button = gr.Button("Add Item", visible=True)
                item_add_button.click(
                    fn=add_list_row,
                    inputs=item_table,
                    outputs=item_table,
                )

                seed_input = gr.Number(label="Random Seed", value=3, precision=0)
                run_button = gr.Button("Run")

                with gr.Row():
                    with gr.Column():
                        prefix_table = gr.List(label="Prefix CLIP", headers=["Prefix"], col_count=1, value=[])
                        prefix_add_button = gr.Button("Add Prefix", visible=True)
                        prefix_add_button.click(
                            fn=add_list_row,
                            inputs=prefix_table,
                            outputs=prefix_table,
                        )
                    with gr.Column():
                        postfix_table = gr.List(label="Postfix CLIP", headers=["Postfix"], col_count=1, value=[])
                        postfix_add_button = gr.Button("Add Postfix", visible=True)
                        postfix_add_button.click(
                            fn=add_list_row,
                            inputs=postfix_table,
                            outputs=postfix_table,
                        )

                yolo_table = gr.List(label="YOLO Items/Prompts", headers=["YOLO Item"], col_count=1, value=[])
                yolo_add_button = gr.Button("Add YOLO Item", visible=True)
                yolo_add_button.click(
                    fn=add_list_row,
                    inputs=yolo_table,
                    outputs=yolo_table,
                )

            with gr.Column():
                output_image = gr.Image(label="Output Image", type="pil")
                output_text = gr.Textbox(label="Output Text", lines=10)

        run_button.click(
            fn=run_segmentation_and_classification,
            inputs=[args_state, input_image, prefix_table, item_table, postfix_table, yolo_table, seed_input],
            outputs=[output_image, output_text],
        )

    return ui


def main(args: argparse.Namespace):
    sam_model_name = args.sam_model if args.sam_model else "sam2.1_hiera_large"
    sam_config_name = args.sam_config if args.sam_config else "sam2.1_hiera_l"
    clip_model_name = args.clip_model if args.clip_model else "ViT-L/14@336px"
    yolo_model_name = args.yolo_model if getattr(args, "yolo_model", "") else "yolo11x-seg.pt"

    Definitions.setup(
        sam_model_name=sam_model_name,
        sam_config_name=sam_config_name,
        clip_model_name=clip_model_name,
        yolo_model_name=yolo_model_name,
        prefixes=[],
        items=[],
        postfixes=[],
        yolo_prompts=[],
    )

    ui = create_interface(args)
    ui.launch(share=args.share)


if __name__ == "__main__":
    parse_args = argparse.ArgumentParser(description="Run SAM2 with CLIP for image segmentation and classification.")
    parse_args.add_argument("--share", action="store_true", help="Share the Gradio app publicly")
    parse_args.add_argument("--no_sam", action="store_true", help="Disables SAM")
    parse_args.add_argument("--sam_model", default="sam2.1_hiera_large", help="SAM model checkpoint path")
    parse_args.add_argument("--sam_config", default="sam2.1_hiera_l", help="SAM config name")
    parse_args.add_argument("--clip_model", default="ViT-L/14@336px", help="CLIP model name")
    parse_args.add_argument("--positive_scale_pin", default=0.0, type=float, help="Positive scale pin for SAM2")
    parse_args.add_argument("--negative_scale_pin", default=0.0, type=float, help="Negative scale pin for SAM2")
    parse_args.add_argument("--yolo", action="store_true", help="Use the YOLO model as the first inference")
    parse_args.add_argument("--require_yolo", action="store_true", help="Require YOLO to pass to next layer")
    parse_args.add_argument("--post_process_yolo", action="store_true", help="Run YOLO post-processing after SAM2")
    parse_args.add_argument("--yolo_model", default="yolo11x-seg.pt", help="YOLO model checkpoint path")

    main(args=parse_args.parse_args())