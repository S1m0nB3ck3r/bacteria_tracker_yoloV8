#!/usr/bin/env python

# Standard imports
import argparse
import shutil
import os
import sys

# External imports
import numpy as np
from sklearn.model_selection import train_test_split


def is_image(f):
    path, ext = os.path.splitext(f)
    candidates = ["jpg", "JPG", "jpeg", "png", "PNG"]
    return ext in candidates


def is_labelled_image(folder, f):
    path, ext = os.path.splitext(f)
    # print("%s -> '%s','%s'"%(f,path,ext))
    candidates = [".jpg", ".JPG", ".jpeg", ".png", ".PNG"]
    if ext not in candidates:
        return (f, f, False)
    if os.path.exists(os.path.join(folder, path + ".txt")):
        return (f, path + ".txt", True)
    return (f, f, False)


def splitds(sourcedir, destdir, seed):
    list_files = [is_labelled_image(sourcedir, f) for f in os.listdir(sourcedir)]
    print("Detected %d files" % len(list_files))
    list_files = [(f, t) for (f, t, ok) in list_files if ok]
    print("Detected %d image,label pairs" % len(list_files))
    list_X = [f for (f, t) in list_files]
    list_Y = [t for (f, t) in list_files]
    X_tv, X_test, Y_tv, Y_test = train_test_split(
        list_X, list_Y, test_size=0.2, random_state=seed
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tv, Y_tv, test_size=0.2, random_state=seed
    )
    os.makedirs(os.path.join(destdir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(destdir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(destdir, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(destdir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(destdir, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(destdir, "labels", "test"), exist_ok=True)
    for f, t in zip(X_train, Y_train):
        shutil.copy(
            os.path.join(sourcedir, f), os.path.join(destdir, "images", "train")
        )
        shutil.copy(
            os.path.join(sourcedir, t), os.path.join(destdir, "labels", "train")
        )
    for f, t in zip(X_val, Y_val):
        shutil.copy(os.path.join(sourcedir, f), os.path.join(destdir, "images", "val"))
        shutil.copy(os.path.join(sourcedir, t), os.path.join(destdir, "labels", "val"))
    for f, t in zip(X_test, Y_test):
        shutil.copy(os.path.join(sourcedir, f), os.path.join(destdir, "images", "test"))
        shutil.copy(os.path.join(sourcedir, t), os.path.join(destdir, "labels", "test"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, help="Path to the source directory")
    parser.add_argument(
        "--dest_dir", type=str, help="Path to the destination directory"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path_in = args.source_dir
    path_out = args.dest_dir

    splitds(path_in, path_out, args.seed)
