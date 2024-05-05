#!/usr/bin/env python
import os
import pickle

import numpy as np

DATA_DIR = "./data/"


def load_cifar_10_train():
    # Pre-allocate memory
    num_samples = 50000  # Total number of samples in CIFAR-10 training set
    image_size = 3072  # Size of each image (32x32x3)
    X = np.empty((num_samples, image_size), dtype=np.uint8)
    y = np.empty(num_samples, dtype=np.uint8)
    index = 0  # Index to keep track of the current position to append
    for file_name in os.listdir(DATA_DIR):
        if file_name.startswith("data_batch_"):
            file_path = os.path.join(DATA_DIR, file_name)
            with open(file_path, "rb") as file:
                batch_data = pickle.load(file, encoding="bytes")
                batch_images = batch_data[b"data"]
                batch_labels = np.array(batch_data[b"labels"], dtype=np.uint8)
                batch_size = len(batch_labels)
                X[index : index + batch_size] = batch_images
                y[index : index + batch_size] = batch_labels
                index += batch_size
                print("Added", file_path)
    return X, y


def load_cifar_10_test():
    # Pre-allocate memory
    num_samples = 10000  # Total number of samples in CIFAR-10 test set
    image_size = 3072  # Size of each image (32x32x3)
    X = np.empty((num_samples, image_size), dtype=np.uint8)
    y = np.empty(num_samples, dtype=np.uint8)
    file_path = os.path.join(DATA_DIR, "test_batch")
    with open(file_path, "rb") as file:
        batch_data = pickle.load(file, encoding="bytes")
        batch_images = batch_data[b"data"]
        batch_labels = np.array(batch_data[b"labels"], dtype=np.uint8)
        batch_size = len(batch_labels)
        X[:batch_size] = batch_images
        y[:batch_size] = batch_labels
    return X, y
