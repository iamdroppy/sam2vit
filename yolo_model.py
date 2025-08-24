from config import Config
from loguru import logger
from PIL import Image
from ultralytics import YOLO

class YoloModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = YOLO(self.config.yolo_model, task="segment")

    def predict(self, image: Image):
        prompts = self.config.yolo_prompts
        try:
            results = self.model(image)

            results = list(results)
            reordered_results = list()
            
            results = [res for res in results if getattr(res, "boxes", None) is not None]
            results = sorted(results, key=lambda x: (getattr(x.boxes, "xyxy", None)[:, 2] - getattr(x.boxes, "xyxy", None)[:, 0]).max() if getattr(x.boxes, "xyxy", None) is not None else 0, reverse=True)
            results = list(results)
            for res in results:
                try:
                    boxes = getattr(res, "boxes", None)
                    if boxes is None:
                        # nothing to do
                        continue

                    # xyxy: Nx4 tensor/array
                    xyxy = getattr(boxes, "xyxy", None)
                    confs = getattr(boxes, "conf", None) or getattr(boxes, "confidence", None)
                    cls_ids = getattr(boxes, "cls", None) or getattr(boxes, "class_id", None)

                    # convert tensors to numpy arrays if necessary
                    if xyxy is None:
                        continue
                    if hasattr(xyxy, "cpu"):
                        xyxy = xyxy.cpu().numpy()
                    if confs is not None and hasattr(confs, "cpu"):
                        confs = confs.cpu().numpy()
                    if cls_ids is not None and hasattr(cls_ids, "cpu"):
                        cls_ids =    cls_ids.cpu().numpy()

                    # Try to get names mapping
                    names = {}
                    if hasattr(self.model, "model") and hasattr(self.model.model, "names"):
                        names = self.model.model.names
                    elif hasattr(res, "names"):
                        names = res.names

                    # Iterate detections
                    for idx, box in enumerate(xyxy):
                        try:
                            x1, y1, x2, y2 = map(int, box.tolist())
                        except Exception:
                            # fallback if box is already a list/tuple
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                        cls_id = int(cls_ids[idx]) if cls_ids is not None else None
                        label = None
                        if isinstance(names, dict) and cls_id is not None:
                            label = names.get(cls_id, str(cls_id))
                        elif isinstance(names, (list, tuple)) and cls_id is not None:
                            label = names[cls_id]
                        else:
                            label = str(cls_id)

                        if label in prompts:
                            conf = float(confs[idx]) if confs is not None else None
                            cropped = image.crop((x1, y1, x2, y2))
                            return cropped, label, conf
                        else:
                            logger.debug(f"YOLO detected: {label} but not in prompts.")
                            return None, None, None
                except Exception as e:
                    logger.debug(f"Failed to parse YOLO result: {e}")
                    continue

            return None, None, None
        except Exception as e:
            logger.warning(f"Failed to process YOLO result: {e}")
            return None, None, None