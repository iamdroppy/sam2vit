import clip
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
class ClipModel:
    def __init__(self, model_name, device):
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.device = device
        
    def process(self, image, prompts):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        top_prompt_index = np.argmax(probs)
        #print(f"Top prompt: {prompts[top_prompt_index]} with probability {probs[0][top_prompt_index]:.4f}")
        return {
            "prompt": prompts[top_prompt_index],
            "probability": probs[0][top_prompt_index],
            "probability_percentage": probs[0][top_prompt_index] * 100
        }