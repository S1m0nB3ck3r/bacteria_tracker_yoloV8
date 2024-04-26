import argparse
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
from my_kalman import main_tracker


def save_label(results, file_path):

    objects = result.object_prediction_list

    with open(file_path, "a") as file:
        for object in objects:
            type = 0
            left = object.bbox.minx / result.image_width
            right = object.bbox.maxx / result.image_width
            top = object.bbox.miny / result.image_height
            bottom = object.bbox.maxy / result.image_height
            file.write(f"{type} {left} {top} {right} {bottom}\n")


def save_image_with_bounding_boxes(save_dir, yolo_result):

    image_name_without_extension = result.path.split("\\")[-1].split(".")[0]
    extension = result.path.split(".")[-1]
    new_image_name = image_name_without_extension + "_annot_." + extension
    new_image_path = os.path.join(save_dir, new_image_name)

    img = Image.open(result.path)
    canvas = img.copy()
    # fnt = ImageFont.truetype("arial.ttf", 24)
    # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 24)
    draw = ImageDraw.Draw(canvas)

    bounding_boxes = yolo_result.boxes.cpu().xyxy.numpy()

    if bounding_boxes.size != 0:
        for bbox in bounding_boxes:
            # x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

            draw.rectangle(bbox, None, (0, 255, 0), 2)

        canvas.save(new_image_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_directory", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, default="./result")
    args = parser.parse_args()

    image_extension = "jpg"

    detection_directory = args.detection_directory

    # Liste pour stocker les sous-répertoires
    detection_directories = []

    # Parcours de tous les éléments du répertoire parent
    for ddir in os.listdir(detection_directory):
        # Chemin complet de l'élément
        path_dir = os.path.join(detection_directory, ddir)
        # Vérification si l'élément est un répertoire
        if os.path.isdir(path_dir):
            detection_directories.append(path_dir)

    weight_path = args.weight_path
    result_path = args.result_path

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # model = YOLO(weight_path)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=weight_path,
        confidence_threshold=0.3,
        device="cuda:0",
    )

    for film_dir in detection_directories:

        print("start detection " + film_dir)

        images_path = [
            os.path.join(film_dir, f)
            for f in os.listdir(film_dir)
            if f.split(".")[-1] == image_extension
        ]

        exp_dir = os.path.join(result_path, os.path.split(film_dir)[-1])
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        print(f"Will save the results into {exp_dir}")

        path_result_label = os.path.join(exp_dir, "labels")
        if not os.path.exists(path_result_label):
            os.mkdir(path_result_label)

        path_result_images_filtered = os.path.join(exp_dir, "filtered")
        if not os.path.exists(path_result_images_filtered):
            os.mkdir(path_result_images_filtered)

        if not os.path.exists(path_result_images_filtered + "\\data"):
            os.mkdir(path_result_images_filtered + "\\data")

        if not os.path.isdir(os.path.join(path_result_images_filtered, "images")):
            os.mkdir(os.path.join(path_result_images_filtered, "images"))

        for i, image_path in enumerate(images_path):
            result = get_sliced_prediction(
                image_path,
                detection_model,
                slice_height=256,
                slice_width=256,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                auto_slice_resolution=False,
            )

            result.export_visuals(
                export_dir=exp_dir,
                file_name=str(i) + ".jpg",
                hide_conf=True,
                hide_labels=True,
            )

            objects = result.object_prediction_list

            print("nb bacteries : " + str(len(objects)))
            image_name_without_extension = image_path.split("\\")[-1].split(".")[0]
            image_extension = image_path.split(".")[-1]
            label_save_path = os.path.join(
                path_result_label, image_name_without_extension + ".txt"
            )

            save_label(result, label_save_path)

        main_tracker(path_result_label, film_dir, path_result_images_filtered)
