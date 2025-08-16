import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from console import Console

pwd = os.path.dirname(os.path.abspath(__file__))
def get_checkpoint_path(model_name="sam2.1_hiera_large"):
    return os.path.join(pwd, "checkpoints", model_name + ".pt")
def get_model_cfg_path(config_name="sam2.1_hiera_l"):
    return os.path.join(pwd, "sam2", "configs", "sam2.1", config_name + ".yaml")

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
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
        
def show_plot_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    

def predict_sam2(image: Image.Image, sam2_model, mask_threshold=0.05, show:bool = False):
    image = np.array(image.convert("RGB"))
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)

    # get the point in the center of the image
    h, w = image.shape[:2]
    point_coords = np.array([[w // 2, h // 2]])
    input_labels = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=input_labels,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    
    h, w = image.shape[:2]
    m = int(min(w, h) * mask_threshold)
    point_coords = np.array([[w//2, h//2], [w+m, h+m], [w-m, h+m]])
    # input_labels
    input_labels = np.array([1, 1, 1])

    mask_input = logits[np.argmax(scores), :, :]
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=input_labels,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    # show_masks(image, masks, scores, point_coords=point_coords, input_labels=input_labels, borders=True)
    
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)
    # remove non-masked areas as transparent
    combined_mask = combined_mask.astype(np.uint8) * 255
    # crop white areas from mask into the image, so what's black is not displayed
    image_with_mask = image.copy()
    image_with_mask[combined_mask == 0] = 255
    image_with_mask = Image.fromarray(image_with_mask)
    # to image
    image_with_mask = image_with_mask.convert("RGBA")

    if show:
        show_plot_image(image_with_mask)
    
    return image_with_mask