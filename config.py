import json
from typing import Dict, Any, Optional, Sequence, List
from loguru import logger
class YoloConfig:
    def __init__(self, yolo: Dict[str, Any]):
        self.enabled = yolo.get("enabled", True)
        self.mode = yolo.get("mode", "output")
        self.model = yolo.get("model", "yolov5s")
        self.confidence_threshold = yolo.get("confidence_threshold", 0.5)
    
    def enabled(self) -> bool:
        return self.enabled
    
    def mode(self) -> str:
        return self.mode

    def model(self) -> str:
        return self.model

    def confidence_threshold(self) -> float:
        return self.confidence_threshold

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.prefixes = config_dict.get("prefixes", [])
        self.items = config_dict.get("items", [])
        self.postfixes = config_dict.get("postfixes", [])
        self.yolo = YoloConfig(config_dict.get("yolo", {"enabled": False, "mode": "output", "model": "yolov5s", "confidence_threshold": 0.5}))

    def get_prefixes(self) -> Sequence[str]:
        return self.prefixes

    def get_items(self) -> Sequence[str]:
        return self.items

    def get_postfixes(self) -> Sequence[str]:
        return self.postfixes

    
    def get_prompts(self) -> List[str]:
        """
        Generate a list of prompt strings by combining each item with every prefix and postfix.

        Returns:
            List[str]: All combinations of prefix + item + postfix.
        """
        prompts: List[str] = []
        for item in self.get_items():
            for prefix in self.get_prefixes():
                for postfix in self.get_postfixes():
                    prompts.append(f"{prefix.strip()} {item.strip()} {postfix.strip()}")
        
        logger.trace(f"Generated {len(prompts)} prompts from config.")
        return prompts
    
    def get_item_from_list(self, prompt: str) -> Optional[str]:
        """
        Extracts the first item from a prompt string.
        Args:
            prompt (str): 
                A string that may contain an item name.
        """
        for item in self.get_items():
            if " " + item + " " in prompt:
                return item
        return None
    
def load_config(file_path: str) -> Config:
    """
    Loads the configuration from a JSON file.
    Args:
        file_path (str): The path to the JSON configuration file.
    Returns:
    Dict[str, Any]: The loaded configuration as a dictionary.
    """
    config = None
    with open(file_path, "r") as f:
        config = Config(json.load(f))
    return config