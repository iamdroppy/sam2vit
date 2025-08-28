import json
from typing import Dict, Any, Optional, Sequence, List
from loguru import logger

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.csv_prefixes = config_dict.get("csv_prefixes", [])
        self.csv_postfixes = config_dict.get("csv_postfixes", [])
        self.prefixes = config_dict.get("prefixes", [])
        self.items = config_dict.get("items", [])
        self.postfixes = config_dict.get("postfixes", [])
        self.yolo_model = config_dict.get("yolo_model", "yolo11x-seg.pt")
        self.yolo_confidence_threshold = config_dict.get("yolo_confidence_threshold", 0.5)
        self.yolo_prompts = config_dict.get("yolo_prompts", [])

    def get_prefixes(self) -> Sequence[str]:
        return self.prefixes

    def get_items(self) -> Sequence[str]:
        return self.items

    def get_postfixes(self) -> Sequence[str]:
        return self.postfixes

    def get_csv_prefixes(self) -> Sequence[str]:
        return self.csv_prefixes

    def get_csv_postfixes(self) -> Sequence[str]:
        return self.csv_postfixes

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

# not used yet.
class ArgsConfig:
    def __init__(self, args: Dict[str, Any]):
        self.args = args

    def get(self, key: str, default: Any = None) -> Any:
        return self.args.get(key, default)
    
    def no_sam(self) -> bool:
        return self.args.get("no_sam", False)
    
    def sam_model(self) -> str:
        return self.args.get("sam_model", "sam2.1_hiera_large")
    
    def sam_config(self) -> str:
        return self.args.get("sam_config", "sam2.1_hiera_l")

    def clip_model(self) -> str:
        return self.args.get("clip_model", "ViT-L/14@336px")

    def use_yolo_model(self) -> bool:
        return self.args.get("use_yolo_model", False)

    def seed(self) -> int:
        return self.args.get("seed", 3)

    def show_image(self) -> bool:
        return self.args.get("show_image", False)

    def input_dir(self) -> str:
        return self.args.get("input_dir", "_input")

    def output_dir(self) -> str:
        return self.args.get("output_dir", "_output")

    def output_original(self) -> bool:
        return self.args.get("output_original", False)

    def yolo(self) -> bool:
        return self.args.get("yolo", False)

    def device(self) -> str:
        return self.args.get("device", "cuda")

    def log_level(self) -> str:
        return self.args.get("log_level", "INFO")

    def file_log_level(self) -> str:
        return self.args.get("file_log_level", "TRACE")

    def file_log_name(self) -> str:
        return self.args.get("file_log_name", "app.log")

    def file_log_rotation(self) -> str:
        return self.args.get("file_log_rotation", "100 MB")

    def file_log_no_reset(self) -> bool:
        return self.args.get("file_log_no_reset", False)

    def positive_scale_pin(self) -> float:
        return self.args.get("positive_scale_pin", 30.0)

    def negative_scale_pin(self) -> float:
        return self.args.get("negative_scale_pin", 0.0)

if __name__ == "__main__":
    pass