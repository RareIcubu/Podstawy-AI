# plik: utils.py
import numpy as np
import gzip
import os

def load_mnist_images(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Brak pliku: {filename}")
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Brak pliku: {filename}")
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # One-hot encoding
    n = data.shape[0]
    one_hot = np.zeros((n, 10), dtype=np.float32)
    for i, digit in enumerate(data):
        one_hot[i, digit] = 1.0
    return one_hot
