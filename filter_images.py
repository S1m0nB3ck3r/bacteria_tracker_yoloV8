from PIL import Image, ImageFilter
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
import os
import cv2

image_IN_directory = "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\global_dataset_filtered\\image_cleaned_RGB"
image_OUT_directory = "C:\\TRAVAIL\\developpement\\bacteria_tracker-main\\global_dataset_filtered\\remove_CC"
image_extension_IN = "jpg"
image_extension_OUT = "jpg"

#for all image in directory
fichiers = [f for f in os.listdir(image_IN_directory) if os.path.isfile(os.path.join(image_IN_directory, f))]

if not os.path.exists(image_OUT_directory):
    os.mkdir(image_OUT_directory)

nb_im = len(fichiers)
i = 0

for f in fichiers:
    if f.split(".")[-1] == image_extension_IN :
        i+=1
        print( str(i) + "/" + str(nb_im))
        print("open image " + str(f))
        
        #filtrage
        image_name = f.split(".")[0]
        img_IN = Image.open(os.path.join(image_IN_directory, f))
        img_mono= img_IN.convert('L')
        img_np_IN = cp.array(img_mono)

        size_x = cp.shape(img_np_IN)[0]
        size_y = cp.shape(img_np_IN)[1]
        x_center = float(size_x) / 2.0
        y_center = float(size_y) / 2.0
        low_pass_size = 3.0  #pixels
        hp_pass_size = 1024

        #mask creation
        x_line = cp.arange(- float(size_x) / 2.0, float(size_x) / 2.0, 1.0, dtype=np.float32)
        y_line = cp.arange(- float(size_y) / 2.0, float(size_y) / 2.0, 1.0, dtype=np.float32)
        y_, x_ = cp.meshgrid(y_line, x_line)
        distance_matrix = cp.sqrt(cp.square(x_)+np.square(y_))
        mask_filter_lp = distance_matrix <= low_pass_size
        # mask_filter_hp = distance_matrix >= hp_pass_size

        FFT_img = cp.fft.fftshift(cp.fft.fft2(img_np_IN))

        cp.putmask(FFT_img, mask_filter_lp, 0+0j)
        # cp.putmask(FFT_img, mask_filter_hp, 0+0j)

        fliterd_image = cp.abs(cp.fft.ifft2(cp.fft.ifftshift(FFT_img)))
        min = fliterd_image.min()
        max = fliterd_image.max()

        img_np_OUT = (255.0 * (fliterd_image - min) / (max - min)).get()

        img_np_OUT = img_np_OUT.astype(np.uint8)
        
        img_OUT = Image.fromarray(img_np_OUT, 'L')
        path_out_f = image_OUT_directory + "\\" + str(image_name) + "." + image_extension_OUT

        img_OUT.save(path_out_f)
        print("save image " + str(f))

        

