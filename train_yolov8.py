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

# start_weight_path = "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\test yolo v8\\best_from_Mehran.pt"
# output_weight_path = (
#     "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\test yolo v8\\best_best.pt"
# )
# data_file = (
#     "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\test yolo v8\\2bacteria.yaml"
# )
# project_path = (
#     "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\test yolo v8\\result_train"
# )


def train(args):
    # Initialize a wandb run
    wandb.init(project="object-detection-bdd")

    # Create the model
    model = YOLO(args.start_weight_path)

    # Register the wandb callback
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
        workers=8,
    )

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        help="Path to the data yaml file for Yolo training",
    )
    parser.add_argument(
        "--start_weight_path",
        type=str,
        default="yolov8n.pt",
        help="Path to the starting weight file",
    )
    parser.add_argument("--project_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
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
    args = parser.parse_args()

    if not os.path.exists(args.project_path):
        os.mkdir(args.project_path)
    train(args)
