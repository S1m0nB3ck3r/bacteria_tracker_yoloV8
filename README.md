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

## Links

- [Original project](https://github.com/Valentin-42/bacteria_tracker)
