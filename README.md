# bacteria_tracker_yoloV8

## Setup

You need to define a `.env` file containing your labelbbox API key and project id.

`
API_KEY=....
PROJECT_ID=...
`

For running the code, you need to create a virtual environment :

```
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset preparation

Before training, we download and generate a dataset with tiles of size $256 \times 256$. The dataset preparation is
performed by running :

```
python download_images_labels_and_cut.py /tmp/Download /tmp/Split
```

This will download the data into `/tmp/Download/`, performs the split in train/valid/test folds into `/tmp/Split`.

## Training

You have to fill in the `2bacteria.yaml` script and to adapt with respect to your data paths.

Then you can run training with various model sizes using a `--start_weight_path` from the list provided by Yolov8 :
`yolov8n.pt, ..., yolov8x.pt` from the smallest to the largest models. See the [models section of the detection
task](https://docs.ultralytics.com/tasks/detect/).

Depending on your GPU RAM available, you may need to adapt the batch size. For example

```
python train_yolov8.py --project_path projects --data_file 2bacteria.yaml --batch 16 --start_weight_path yolov8x.pt
```

If you log with wandb, you need more GPU ram. Running `yolov8x` with a batch size of $16$, and logging to wandb requires
a peak GPU RAM usage of almost 8GB.

You can use autobatch size by keeping the default `batch=-1` but this does not work if you use wandb. It will adapt the
batch size as if wandb is not used but the use of wandb involves more the GPU RAM.


## Links

- [Original project](https://github.com/Valentin-42/bacteria_tracker)
