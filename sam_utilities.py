import os
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Optional visualization deps
try:  # matplotlib is only needed for visualization helpers
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

try:  # OpenCV is only needed for border drawing in masks
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

from sam2.sam2_image_predictor import SAM2ImagePredictor

pwd = os.path.dirname(os.path.abspath(__file__))


def get_checkpoint_path(model_name: str = "sam2.1_hiera_large") -> str:
    """Return the absolute path to a SAM2 checkpoint file for the given model name."""
    return os.path.join(pwd, "checkpoints", model_name + ".pt")


def get_model_cfg_path(config_name: str = "sam2.1_hiera_l") -> str:
    """Return the absolute path to a SAM2 config YAML for the given config name."""
    return os.path.join(pwd, "sam2", "configs", "sam2.1", config_name + ".yaml")

def show_mask(mask: np.ndarray, ax: Any, random_color: bool = False, borders: bool = True) -> None:
    """Overlay a mask on a Matplotlib axis, optionally with borders."""
    if plt is None:
        raise ImportError("matplotlib is required for show_mask() but is not installed.")
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        if cv2 is None:
            raise ImportError("opencv-python is required for drawing borders but is not installed.")
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords: np.ndarray, labels: np.ndarray, ax: Any, marker_size: int = 375) -> None:
    """Plot positive and negative points on a Matplotlib axis."""
    if plt is None:
        raise ImportError("matplotlib is required for show_points() but is not installed.")
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box: Sequence[int], ax: Any) -> None:
    """Draw a bounding box given [x0, y0, x1, y1] on a Matplotlib axis."""
    if plt is None:
        raise ImportError("matplotlib is required for show_box() but is not installed.")
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(
    image: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    point_coords: Optional[np.ndarray] = None,
    box_coords: Optional[Sequence[int]] = None,
    input_labels: Optional[np.ndarray] = None,
    borders: bool = True,
) -> None:
    """Visualize masks and optional prompts on the provided image."""
    if plt is None:
        raise ImportError("matplotlib is required for show_masks() but is not installed.")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None 
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        
def show_plot_image(image: Image.Image) -> None:
    """Display a PIL image using Matplotlib without axes."""
    if plt is None:
        raise ImportError("matplotlib is required for show_plot_image() but is not installed.")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    

def _combine_masks_to_alpha(masks: np.ndarray) -> np.ndarray:
    """Combine a stack of boolean masks into a single 0/255 uint8 alpha mask."""
    # Equivalent to iterative logical_or but vectorized
    combined = np.any(masks, axis=0)
    return combined.astype(np.uint8) * 255


def predict_sam2(
    image: Image.Image,
    sam2_model: Any,
    mask_threshold: float = 0.05,
    show: bool = False,
) -> Image.Image:
    """Run SAM2 predictor on an image and return an RGBA image with background whitened.

    Args:
        image: PIL image to process.
        sam2_model: Initialized SAM2 model instance.
        mask_threshold: Fraction of the shorter image side used to sample additional points.
        show: If True, display the resulting image with Matplotlib.
    Returns:
        An RGBA PIL Image where non-masked areas are white.
    """
    # Convert input PIL image to a 3-channel RGB numpy array (H, W, 3)
    image_np = np.array(image.convert("RGB"))
    # Initialize the SAM2 predictor and bind the image
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_np)

    # Seed with a single positive point at the image center
    h, w = image_np.shape[:2]
    point_coords = np.array([[w // 2, h // 2]])
    input_labels = np.array([1])

    # First pass: request multiple candidate masks and their scores/logits
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=input_labels,
        multimask_output=True,
    )
    # Sort candidates by confidence (descending) and reorder outputs
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    
    # Define additional positive points around the center for refinement
    m = int(min(w, h) * mask_threshold)
    point_coords = np.array([[w//2, h//2], [w+m, h+m], [w-m, h+m]])
    input_labels = np.array([1, 1, 1])

    # Use the top candidate's logits as a prior and refine to a single mask
    mask_input = logits[np.argmax(scores), :, :]
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=input_labels,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    # Merge mask(s) into a single alpha mask and whiten background pixels
    combined_mask = _combine_masks_to_alpha(masks)
    image_with_mask = image_np.copy()
    image_with_mask[combined_mask == 0] = 255
    image_with_mask = Image.fromarray(image_with_mask)
    image_with_mask = image_with_mask.convert("RGBA")

    # Optionally display the result
    if show:
        show_plot_image(image_with_mask)
    
    return image_with_mask