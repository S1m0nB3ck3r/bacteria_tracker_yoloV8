# coding: utf-8

"""
This script is used to train a YOLOv8 model on the bacteria dataset
"""

# Standard imports
import argparse
import datetime
import os

# External imports
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback


def train(args):
    # Initialize a wandb run
    log_wandb = args.wandb_project is not None and args.wandb_entity is not None
    if log_wandb:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)

    # Create the model
    model = YOLO(args.start_weight_path)

    # Register the wandb callback
    if log_wandb:
        add_wandb_callback(model, enable_model_checkpointing=True)

    cur_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    projet_name = f"BacteriaYoloV8_{cur_datetime}"

    print("start training")
    results = model.train(
        model=args.start_weight_path,
        data=args.data_file,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project_path,
        name=projet_name,
        verbose=True,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        weight_decay=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment=None,
        workers=8,
    )

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        help="Path to the data yaml file for Yolo training",
        required=True,
    )
    parser.add_argument(
        "--start_weight_path",
        type=str,
        default="yolov8n.pt",
        help="Path to the starting weight file",
    )
    parser.add_argument("--project_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch", type=int, default=-1
    )  # Default: -1 for autobatch size
    parser.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="Every input image is resized to this size",
    )
    parser.add_argument("--max_det", type=int, default=2000)
    parser.add_argument("--device", type=int, default=0)  # 0 : GPU
    parser.add_argument("--wandb_entity", type=str, default=None)  # 0 : GPU
    parser.add_argument("--wandb_project", type=str, default=None)  # 0 : GPU
    args = parser.parse_args()

    if not os.path.exists(args.project_path):
        os.mkdir(args.project_path)
    train(args)
