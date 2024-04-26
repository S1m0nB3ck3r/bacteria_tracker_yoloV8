import os
import pandas as pd
import numpy as np
from matplotlib  import pyplot as plt
from PIL import Image
import cv2


def convert_to_avi(dir_images, extension, avi_path):
    images = [i for i in os.listdir(dir_images) if i.endswith(extension)]

    frame = cv2.imread(os.path.join(dir_images, images[0]))
    height, width, layers  = frame.shape

    video = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))  

    for i in images:
        print("writting ", i)
        frame = cv2.imread(os.path.join(dir_images, i))
        video.write(frame)

    video.release()

#repertoires et chemins
base_path = r'D:\bacteria_tracker_yoloV8'

analyse_directories =[
    'result_reduced',
    'result_full_confidence_02'
]

extension = ".jpg"

"D:\bacteria_tracker_yoloV8\result_best_reduced_dataset_yolov8x\2022_12_14\filtered\images"

for dir in analyse_directories:
    path_dir = os.path.join(base_path, dir)
    dirs = os.listdir(path_dir)
    sub_directories = []
    # for d in dirs:
    #     if os.path.isdir(os.path.join(path_dir, d)):
    #         sub_directories.append(d)

    d = []
    sub_directories = [d for d in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir,d))]

    for sub_dir in sub_directories:
        dir_images = os.path.join(base_path, dir, sub_dir, "filtered", "images")
        video_name = dir + "_" + sub_dir + ".avi"
        video_path = os.path.join(dir, video_name)
        print("start converting ", dir_images)
        convert_to_avi(dir_images, extension, video_path)


