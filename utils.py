import argparse
def get_args():
    
    default_sam_model_name = "sam2.1_hiera_large"
    default_sam_config_name = "sam2.1_hiera_l"
    default_clip_model_name = "ViT-L/14@336px"
    default_seed = 3
    
    arg_parsed = argparse.ArgumentParser(description="Run SAM2 with CLIP for image segmentation and classification.")
    
    # defaults
    # SAM2
    arg_parsed.add_argument("--no_sam", "-x", default=False, action="store_true", help="Disables the Segmentation")
    arg_parsed.add_argument("--sam_model", default=default_sam_model_name, type=str, help=f"SAM2 model name (default: {default_sam_model_name})")
    arg_parsed.add_argument("--sam_config", default=default_sam_config_name, type=str, help=f"SAM2 config name (default: {default_sam_config_name})")
    # CLIP
    arg_parsed.add_argument("--clip_model", "-c", default=default_clip_model_name, type=str, help=f"CLIP model name (default: {default_clip_model_name})")

    arg_parsed.add_argument("--use_yolo_model", "-y", default=False, action="store_true", help="Use YOLO model for object detection")
    # np
    arg_parsed.add_argument("--seed", "-S", default=default_seed, type=int, help=f"Random seed for reproducibility (default: {default_seed})")
    arg_parsed.add_argument("--show_image", "-g", action="store_true", default=False, help="Show the output image after each processing step")
    
    # io
    arg_parsed.add_argument("--input_dir", "-i", default="_input", type=str, required=True, help="Path to the dataset directory (default: _input)")
    arg_parsed.add_argument("--output_dir", "-o", default="_output", type=str, required=True, help="Path to the output directory (default: _output)")
    arg_parsed.add_argument("--output_original", default=False, action='store_true', help="If the output should be original, otherwise, segmented (default: False)")

    # logs
    arg_parsed.add_argument("--device", "-d", type=str, default="cuda", choices=["cpu", "cuda"], help="Sets the device")
    arg_parsed.add_argument("--log_level", "-l", type=str, default="INFO", choices=["WORK", "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Sets the log level")
    arg_parsed.add_argument("--file_log_level", "-u", type=str, default="TRACE", choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Sets the log level of `app.log` file (default: trace)")
    arg_parsed.add_argument("--file_log_name", "-w", type=str, default="app.log", help="Sets the log file name (default: app.log)")
    arg_parsed.add_argument("--file_log_rotation", "-r", type=str, default="100 MB", help="Sets the log file rotation size (default: 100 MB)")
    arg_parsed.add_argument("--file_log_no_reset", "-z", type=bool, default=False, help="Do not reset file_log_name on boot")

    arg_parsed.add_argument("--positive_scale_pin", "-p", type=float, default=30, help="Scale pin for positive points in SAM2 model (default: 30)")
    arg_parsed.add_argument("--negative_scale_pin", "-n", type=float, default=0, help="Scale pin for negative points in SAM2 model (default: 0)")
    #arg_parsed.add_argument("--seed", "-s", default=3, type=int, help="Random seed for reproducibility (default: 3)")
    return arg_parsed.parse_args()