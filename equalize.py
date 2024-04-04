#!/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import cv2
import skimage
import numpy as np


def equalize(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    p2, p98 = np.percentile(image, (2, 98))
    equalized_image = skimage.exposure.rescale_intensity(image, in_range=(p2, p98))
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

    return equalized_image


def equalize_dir(source_dir, destination_dir):
    for image_path in os.listdir(source_dir):
        print("Processing", image_path)
        equalized_image = equalize(os.path.join(source_dir, image_path))
        cv2.imwrite(os.path.join(destination_dir, image_path), equalized_image)
        # show_image(image, equalized_image)


def show_image(image, equalized_image):
    cv2.imshow("Original Image", image)
    cv2.imshow("Equalized Image", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python3 equalize.py <source_dir> <destination_dir>")
        sys.exit(1)

    source_dir = sys.argv[1]
    destination_dir = sys.argv[2]
    equalize_dir(source_dir, destination_dir)
