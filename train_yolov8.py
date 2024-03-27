from ultralytics import YOLO
import os

start_weight_path = "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\test yolo v8\\best_from_Mehran.pt"
output_weight_path = "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\test yolo v8\\best_best.pt"
data_file = "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\test yolo v8\\2bacteria.yaml"
project_path = "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\test yolo v8\\result_train"

if __name__ == '__main__':

    if not os.path.exists(project_path):
        os.mkdir(project_path)

    model = YOLO(start_weight_path)

    print("start training")
    results = model.train(model=start_weight_path, data=data_file, epochs=1000, batch=-1,
                          project=project_path, name="test", verbose=True, imgsz=256,
                          max_det=2000, device=0, workers=8)

    print("finished")
