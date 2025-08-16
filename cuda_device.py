import torch
from rich import print
def device_check(device, device_clip):
    if (device.type == "cpu" or device_clip.type == "cpu"):
        print("💾 [yellow][bold]For [underline]SAM2[/underline] or [underline]CLIP[/underline][/bold], it is recommended to use a GPU for better performance. ")
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        print(f"💾 [green][bold]Using [underline]{device.type}[/underline] with [underline]{torch.cuda.get_device_name(0)}[/underline] GPU[/bold][/green]")
        print(f"💾 [green][bold]Using bfloat16[/green][/bold]")
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            print(f"💾 [green][bold]Enabling TF32 for matmul and cuDNN[/bold][/green]")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        print(f"💾 [red][bold]Using [underline]{device.type}[/underline][/bold], which is [red bold underline]NOT SUPPORTED[/red bold underline].")
        exit(0)