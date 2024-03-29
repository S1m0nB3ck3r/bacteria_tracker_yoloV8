from ultralytics import YOLO
from PIL import Image,ImageDraw, ImageFont
import os
import numpy as np
from my_kalman import main_tracker

def save_label(results, file_path):

    with open(file_path, "a") as file:
        for result in results:
            for box in result.boxes:
                type = 0
                left = box.xyxy[0]
                right = box.xyxy[1]
                top = box.xyxy[2]
                bottom = box.xyxy[3]
                file.write(f"{type}\s{left}\{top}\s{right}\s{bottom}\n")

def save_image_with_bounding_boxes(save_dir, yolo_result):

    image_name_without_extension = result.path.split("\\")[-1].split(".")[0]
    extension = result.path.split(".")[-1]
    new_image_name = image_name_without_extension + "_annot_." + extension
    new_image_path = os.path.join(save_dir, new_image_name) 

    img = Image.open(result.path)
    canvas = img.copy()
    fnt = ImageFont.truetype("arial.ttf", 24)
    # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 24)
    draw = ImageDraw.Draw(canvas)

    bounding_boxes = yolo_result.boxes.cpu().xyxy.numpy()

    if bounding_boxes.size != 0:

        for bbox in bounding_boxes:

            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

            draw.rectangle(bbox,None,(0,255,0),2)

        canvas.save(new_image_path)


if __name__ == '__main__' :

    detection_directory = "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\detect2"

    # Liste pour stocker les sous-répertoires
    detection_directories = []

    # Parcours de tous les éléments du répertoire parent
    for dir in os.listdir(detection_directory):
        # Chemin complet de l'élément
        path_dir = os.path.join(detection_directory, dir)
        # Vérification si l'élément est un répertoire
        if os.path.isdir(path_dir):
            detection_directories.append(dir)


    print(detection_directories)

    weight_path= "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\test yolo v8\\best_from_Mehran.pt"
    test_name = "result_from_mehran_train"
    result_path = "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\test yolo v8\\" +  test_name

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    model = YOLO(weight_path)

    for path_image in detection_directories:

        print("start detection " + path_image)

        path_orinal_images = detection_directory + "\\" + path_image #chemin des images à detecter
        
        project_name = path_image
        project_path = result_path

        path_result_label = result_path + "\\" + project_name + "\\labels"


        path_result_images = result_path + "\\" + project_name
        path_result_images_filtered = result_path + "\\" + project_name + "\\images_filtered"

        results = model(path_orinal_images, stream=True)

        for i, result in enumerate(results):
            print("nb bacteries" + str(len(result.boxes)))
            image_name_without_extension = result.path.split("\\")[-1].split(".")[0]
            image_extension = result.path.split(".")[-1]
            label_save_path = os.path.join(path_result_label, image_name_without_extension + ".txt")
            image_save_path = os.path.join(path_result_images, image_name_without_extension + "." + image_extension)

            result.save_txt(label_save_path)
            save_image_with_bounding_boxes(path_result_images, result)

        if not os.path.exists(path_result_images_filtered):
            os.mkdir(path_result_images_filtered)

        if not os.path.exists(path_result_images_filtered + "\\data"):
            os.mkdir(path_result_images_filtered + "\\data")

        if not os.path.isdir(os.path.join(path_result_images_filtered,"images")):
            os.mkdir(os.path.join(path_result_images_filtered,"images"))

        main_tracker(path_result_label, path_orinal_images, path_result_images_filtered)



