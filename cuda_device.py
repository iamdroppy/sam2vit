import torch
from loguru import logger


def _enable_bf16_autocast_for_cuda() -> None:
    """Enable global autocast in BF16 on CUDA (preserves original side effect)."""
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    logger.info("💾 Using bfloat16")


def _enable_tf32_if_supported(device_index: int) -> None:
    """Enable TF32 on Ampere (SM >= 80) and above."""
    props = torch.cuda.get_device_properties(device_index)
    if props.major >= 8:
        logger.info(" --> 💾 Enabling TF32 for matmul and cuDNN")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _get_active_cuda_index(dev: torch.device) -> int:
    """Return the CUDA device index, falling back to current device when None."""
    if dev.index is not None:
        return dev.index
    return torch.cuda.current_device()


def device_check(device: torch.device, device_clip: torch.device) -> None:
    """
    Validate devices and configure CUDA settings.

    Behavior (preserved):
    - Warn if either SAM2 or CLIP is on CPU.
    - On CUDA: enable BF16 autocast and TF32 (if supported), print GPU info.
    - Otherwise: print NOT SUPPORTED and exit(0).
    """
    # Recommend GPU if either component runs on CPU
    if device.type != "cpu" or device_clip.type != "cpu":
        logger.warning(
            " --> 💾 For sam2vit, it is recommended to use a GPU for better performance."
        )

    if device.type == "cuda":
        # Pick the correct CUDA device index (don't assume 0)
        idx = _get_active_cuda_index(device)
        gpu_name = torch.cuda.get_device_name(idx)
        logger.info(
            f" --> 💾 Using {device.type} on {gpu_name} GPU"
        )
        _enable_bf16_autocast_for_cuda()
        _enable_tf32_if_supported(idx)
        return None

    # Anything other than CUDA is considered unsupported (preserve exit behavior)
    logger.error(
        f" --> 💾 Using {device.type} NOT SUPPORTED by sam2vit. Quitting..."
    )
    exit(0)
