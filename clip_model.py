from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import torch

# Optional import: provide a helpful error if unavailable at runtime
try:
    import clip  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    clip = None  # type: ignore


class ClipModel:
    """Thin wrapper around OpenAI CLIP for image-text similarity.

    This class loads a CLIP model and exposes a simple `process` method that
    ranks a list of text prompts against a given image, returning the top match
    and its probability.
    """

    def __init__(self, model_name: str, device: Union[str, torch.device]):
        """Load CLIP and its preprocessing pipeline on the specified device.

        Args:
            model_name: Name of the CLIP variant (e.g., "ViT-L/14@336px").
            device: Device identifier (e.g., torch.device("cuda") or "cpu").
        """
        if clip is None:
            raise ImportError("The 'clip' package (OpenAI CLIP) is required but not installed.")
        
        # Load CLIP model and its default preprocess transform
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.device = device

    def process(self, image: Image.Image, prompts: List[str]) -> Dict[str, Any]:
        """Compute similarity between an image and a set of prompts and pick the best.
        
        Args:
            image: PIL Image to evaluate.
            prompts: Candidate text prompts to compare with the image.

        Returns:
            Dict with keys:
            - prompt: The best-matching prompt string.
            - probability: The probability for the best prompt (0..1).
            - probability_percentage: probability * 100.
        """
        # 1) Preprocess and batch the image (shape becomes [1, C, H, W])
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        # 2) Tokenize prompts into CLIP-compatible tokens
        text_tokens = clip.tokenize(prompts).to(self.device)

        # Disable gradients for inference-only work
        with torch.no_grad():
            # Optionally compute features (kept for potential future use)
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

            # 3) Forward pass to get similarity logits; then softmax to probabilities
            logits_per_image, logits_per_text = self.model(image_tensor, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # 4) Get the index of the highest-probability prompt
            top_prompt_index = int(np.argmax(probs))

            return {
                "prompt": prompts[top_prompt_index],
                "probability": float(probs[0][top_prompt_index]),
                "probability_percentage": float(probs[0][top_prompt_index] * 100),
                "image_features": image_features,
                "text_features": text_features,
                "logits_per_text": logits_per_text
            }