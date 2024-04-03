# coding: utf-8

# Standard imports
from urllib.request import urlopen
import sys
import os

# External imports
from dotenv import load_dotenv
import labelbox as lb
import json
from PIL import Image, ImageDraw, ImageFont
from rectangles import Rectangle

# Local imports
import split_ds


def export(output_folder, API_KEY, project_id):
    """
    Export the images and labels from a Labelbox project and save them to the specified output folder
    This script will also split the images into 256x256 tiles and save them to the output folder
    """
    output_folder_256 = os.path.join(output_folder, "sub256")
    output_folder_annot = os.path.join(output_folder, "annot")
    isExist = os.path.exists(output_folder)
    if not isExist:
        os.makedirs(output_folder)

    isExist = os.path.exists(output_folder_256)
    if not isExist:
        os.makedirs(output_folder_256)

    isExist = os.path.exists(output_folder_annot)
    if not isExist:
        os.makedirs(output_folder_annot)

    client = lb.Client(API_KEY)

    project = client.get_project(project_id)

    labels = project.export_labels(download=True)
    # project.DataRow
    save(labels)
    for label in labels:
        img_url = label["Labeled Data"]
        image_filename = label["External ID"]

        image_fullpath = os.path.join(output_folder, image_filename)
        if os.path.exists(image_fullpath):
            img = Image.open(image_fullpath)
        else:
            img = Image.open(urlopen(img_url))
        bbox = img.size
        if bbox is None:
            print("Skipping '%s'" % image_filename)
            continue
        bbox = 0, 0, bbox[0], bbox[1]
        if not os.path.exists(image_fullpath):
            img.save(image_fullpath)
        split_labels_to_txt_files(label, bbox, output_folder)
        if (bbox[2] > 256) or (bbox[3] > 256):
            image_cutter(label, img, output_folder_256, output_folder_annot)


def save(labels):
    with open("labelbox_export.json", "w") as f:
        json.dump(labels, f)
    print("done")


def split_labels_to_txt_files(data, ibox, out_folder):
    # Extract the image filename from the 'External ID' key
    image_filename = data["External ID"]

    # Remove the file extension from the filename
    image_filename_without_ext = os.path.splitext(image_filename)[0]

    # Create a new file with the same name as the image file
    output_filename = f"{image_filename_without_ext}.txt"
    _, _, iw, ih = ibox
    iw = float(iw)
    ih = float(ih)
    print(ibox)

    # Open the output file in write mode
    with open(os.path.join(out_folder, output_filename), "w") as output_file:
        # Iterate over the objects in the label data
        for obj in data["Label"]["objects"]:
            # Extract the bounding box coordinates from the 'bbox' key
            bbox = obj["bbox"]
            print(bbox)
            x, y, w, h = (
                (bbox["left"] + bbox["width"] / 2.0) / iw,
                (bbox["top"] + bbox["height"] / 2.0) / ih,
                bbox["width"] / iw,
                bbox["height"] / ih,
            )

            # Write the bounding box coordinates to the output file
            print(f"0 {x} {y} {w} {h}\n")
            output_file.write(f"0 {x} {y} {w} {h}\n")

    print("done " + str(image_filename))


def is_valid(rect):
    if rect is None:
        return False
    if rect.area() < 4:
        return False
    if rect.width() <= 5:
        return False
    if rect.height() <= 5:
        return False
    return True


def image_cutter(data, img, output_path_sub, output_path_annot):
    print("Cutting image")
    # Extract the image filename from the 'External ID' key
    image_filename = data["External ID"]

    # Remove the file extension from the filename
    image_filename_without_ext = os.path.splitext(image_filename)[0]

    # Create a new file with the same name as the image file
    # output_filename = f"{image_filename_without_ext}.txt"
    iw, ih = img.size
    # fw = float(iw)
    # fh = float(ih)

    canvas = img.copy()
    fnt = ImageFont.truetype("arial.ttf", 24)
    # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 24)
    draw = ImageDraw.Draw(canvas)
    for i in range(0, ih, 256):
        draw.line([(0, i), (iw, i)], (255, 0, 0), 2)
    for j in range(0, iw, 256):
        draw.line([(j, 0), (j, ih)], (255, 0, 0), 2)
    for i in range(0, ih, 256):
        for j in range(0, iw, 256):
            draw.text(
                (i + 10, j + 10),
                "%d,%d" % (j // 256, i // 256),
                font=fnt,
                fill=(255, 0, 0),
            )
    bboxes = []
    for obj in data["Label"]["objects"]:
        bbox = obj["bbox"]
        x, y, w, h = bbox["left"], bbox["top"], bbox["width"], bbox["height"]
        bboxes.append(Rectangle(x, y, x + w, y + h))
        draw.rectangle([(x, y), (x + w, y + h)], None, (0, 255, 0), 2)
    if output_path_annot is not None:
        canvas.save(
            os.path.join(output_path_annot, image_filename_without_ext + "_annot.jpg")
        )

    # print(bboxes)

    for i in range(0, ih, 256):
        if ih - i < 256:
            continue
        for j in range(0, iw, 256):
            if iw - j < 256:
                continue
            rect = Rectangle(j, i, j + 256, i + 256)
            # print((i,j,rect))

            crop_labels = [(r, r.intersection(rect)) for r in bboxes]
            crop_labels = [r_in for r, r_in in crop_labels if is_valid(r_in)]
            # print(crop_labels)

            if len(crop_labels) == 0:
                continue

            crop = img.crop((j, i, j + 256, i + 256))
            crop.save(
                os.path.join(
                    output_path_sub,
                    image_filename_without_ext + "_%d_%d.jpg" % (i // 256, j // 256),
                )
            )

            draw = ImageDraw.Draw(crop)
            with open(
                os.path.join(
                    output_path_sub,
                    image_filename_without_ext + "_%d_%d.txt" % (i // 256, j // 256),
                ),
                "w",
            ) as f:
                for l in crop_labels:
                    w = l.width() / 256.0
                    h = l.height() / 256.0
                    x = (l.x1 - rect.x1) / 256.0 + w / 2
                    y = (l.y1 - rect.y1) / 256.0 + h / 2
                    f.write(f"0 {x} {y} {w} {h}\n")
                    draw.rectangle(
                        [
                            ((x - w / 2.0) * 256, (y - h / 2.0) * 256),
                            ((x + w / 2.0) * 256, (y + h / 2.0) * 256),
                        ],
                        None,
                        (0, 255, 0),
                        2,
                    )

            if output_path_annot is not None:
                crop.save(
                    os.path.join(
                        output_path_annot,
                        image_filename_without_ext
                        + "_%d_%d_annot.jpg" % (i // 256, j // 256),
                    )
                )


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage : download_label_and_images_v2.py <download_dir> <split_dir")
        sys.exit(-1)

    # Load the API key and project ID from the environment variables
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    project_id = os.getenv("PROJECT_ID")

    # Extract the download folder and split folder from the command line arguments
    dl_folder = sys.argv[1]
    split_folder = sys.argv[2]

    export(dl_folder, API_KEY, project_id)
    split_ds.splitds(dl_folder, split_folder)
