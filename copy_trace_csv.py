import os
import shutil

dir_path = r"D:\bacteria_tracker_yoloV8\result_best_full_dataset_256_yolov8n_confiance_02"

list_dir = [dir for dir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, dir))]

for dir in list_dir:
    path_csv = os.path.join(dir_path, dir, "filtered", "traces.csv")
    new_name = dir + ".csv"
    path_new_name = os.path.join(dir_path, new_name)
    shutil.copyfile(path_csv, path_new_name)