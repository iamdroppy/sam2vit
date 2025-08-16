import torch
from rich import print


def _enable_bf16_autocast_for_cuda() -> None:
    """Enable global autocast in BF16 on CUDA (preserves original side effect)."""
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print("💾 [green][bold]Using bfloat16[/green][/bold]")


def _enable_tf32_if_supported(device_index: int) -> None:
    """Enable TF32 on Ampere (SM >= 80) and above."""
    props = torch.cuda.get_device_properties(device_index)
    if props.major >= 8:
        print("💾 [green][bold]Enabling TF32 for matmul and cuDNN[/bold][/green]")
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
    if device.type == "cpu" or device_clip.type == "cpu":
        print(
            "💾 [yellow][bold]For [underline]SAM2[/underline] or [underline]CLIP[/underline][/bold], it is recommended to use a GPU for better performance. "
        )

    if device.type == "cuda":
        # Pick the correct CUDA device index (don't assume 0)
        idx = _get_active_cuda_index(device)
        gpu_name = torch.cuda.get_device_name(idx)
        print(
            f"💾 [green][bold]Using [underline]{device.type}[/underline] with [underline]{gpu_name}[/underline] GPU[/bold][/green]"
        )
        _enable_bf16_autocast_for_cuda()
        _enable_tf32_if_supported(idx)
        return None

    # Anything other than CUDA is considered unsupported (preserve exit behavior)
    print(
        f"💾 [red][bold]Using [underline]{device.type}[/underline][/bold], which is [red bold underline]NOT SUPPORTED[/red bold underline] by [red]CLIP[/red]."
    )
    exit(0)
