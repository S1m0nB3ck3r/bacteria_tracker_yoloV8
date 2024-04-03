import os
import sys
import glob
import pathlib

from ultralytics import YOLO
from PIL import Image, ImageDraw


def save_image_with_bounding_boxes(save_dir, yolo_result):

    img_path = yolo_result.path
    image_name_without_extension = pathlib.Path(img_path).stem
    new_image_name = image_name_without_extension + "_annot_.png"
    new_image_path = os.path.join(save_dir, new_image_name)

    img = Image.open(img_path)
    canvas = img.copy()
    # fnt = ImageFont.truetype("arial.ttf", 24)
    # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 24)
    draw = ImageDraw.Draw(canvas)

    bounding_boxes = yolo_result.boxes.cpu().xyxy.numpy()

    if bounding_boxes.size != 0:
        for bbox in bounding_boxes:
            draw.rectangle(bbox, None, (0, 255, 0), 2)
        canvas.save(new_image_path)

    img.close()
    canvas.close()


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(f"Usage : {sys.argv[0]} <img_dir> <save_dir> <yolo.pt>")
        sys.exit(-1)

    img_dir = sys.argv[1]
    save_dir = sys.argv[2]
    weight_path = sys.argv[3]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model = YOLO(weight_path)
    results = model(img_dir, stream=True)
    for result in results:
        save_image_with_bounding_boxes(save_dir, result)
