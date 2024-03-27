import labelbox as lb
from PIL import Image,ImageDraw, ImageFont
from urllib.request import urlopen
import os

download_directory = "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\dataset_full"

if not os.path.exists(download_directory):
    os.mkdir(download_directory)

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHE2ZDV3bjUwOGxlMDcwN2NwbzYydGk3Iiwib3JnYW5pemF0aW9uSWQiOiJjbHE2ZDV3bXgwOGxkMDcwNzZyYWFmMDQwIiwiYXBpS2V5SWQiOiJjbHE2cWw3MXMwMDRrMDcxb2NwdTlkMHl1Iiwic2VjcmV0IjoiM2Y2YTQ0Nzk1YjQzYjJmNTY2Y2UzZjRkOTVjOTJiMGEiLCJpYXQiOjE3MDI2NTEyNDIsImV4cCI6MjMzMzgwMzI0Mn0.nUf1HSCzi2a1BE20h_To1g97DKTqRmliQvj3kCG8WE4"
PROJECT_ID = "clryqepub1izg072acvagcbnt"

client = lb.Client(API_KEY)

project = client.get_project(PROJECT_ID)
task = project.export_v2()
task.wait_till_done()

if task.errors:
  print(task.errors)
else:
    print("task :")
    print(task)
    i =0
    for image in task.result:
        #for all images in project
        data_row_id = image['data_row']['external_id']  #image name
        data_row_url = image['data_row']['row_data']    #image url
        print(data_row_id)

        img = Image.open(urlopen(data_row_url))
        size_img = img.getbbox()
        _,_,iw,ih = size_img
        iw=float(iw)
        ih=float(ih)

        if size_img is None:
            print("Skipping '%s'" % data_row_id)
            continue

        image_fullpath = os.path.join(download_directory, data_row_id)
        image_filename_without_ext = os.path.splitext(data_row_id)[0]
        bbox_filename = f"{image_filename_without_ext}.txt"

        if not os.path.exists(image_fullpath):
            img.save(image_fullpath)

        with open(os.path.join(download_directory,bbox_filename), 'w') as output_file:
            for l in image['projects'][PROJECT_ID]["labels"]:
                for obj in l['annotations']['objects']:
                    bbox = obj['bounding_box']
                    # Open the output file in write mode
                    x, y, w, h = (bbox['left']+bbox['width']/2.)/iw, (bbox['top']+bbox['height']/2.)/ih, bbox['width']/iw, bbox['height']/ih
                  
                    # Write the bounding box coordinates to the output file
                    output_file.write(f"0 {x} {y} {w} {h}\n")