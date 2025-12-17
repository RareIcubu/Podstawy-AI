import gzip
import numpy as np
import os

def load_mnist_images(filename):
    """Wczytuje obrazy z formatu idx3-ubyte.gz"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Nie znaleziono pliku: {filename}. Pobierz go!")

    with gzip.open(filename, 'rb') as f:
        # Format IDX3:
        # bajty 0-3: magic number
        # bajty 4-7: liczba obrazów
        # bajty 8-11: liczba wierszy (28)
        # bajty 12-15: liczba kolumn (28)
        # Dlatego pomijamy pierwsze 16 bajtów nagłówka
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    # Przekształcamy na float 0.0-1.0 i formatujemy do kształtu (N, 784)
    data = data.reshape(-1, 784).astype(np.float32) / 255.0
    return data

def load_mnist_labels(filename):
    """Wczytuje etykiety z formatu idx1-ubyte.gz"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Nie znaleziono pliku: {filename}. Pobierz go!")

    with gzip.open(filename, 'rb') as f:
        # Format IDX1:
        # bajty 0-3: magic number
        # bajty 4-7: liczba etykiet
        # Dlatego pomijamy pierwsze 8 bajtów
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    
    # One-hot encoding (zamiana cyfry 5 na [0,0,0,0,0,1,0...])
    num_labels = data.shape[0]
    one_hot = np.zeros((num_labels, 10), dtype=np.float32)
    for i, digit in enumerate(data):
        one_hot[i, digit] = 1.0
        
    return one_hot

def get_training_data():
    print("Wczytywanie zbioru TRENINGOWEGO...")
    X = load_mnist_images('Minst/train-images-idx3-ubyte.gz')
    y = load_mnist_labels('Minst/train-labels-idx1-ubyte.gz')
    return X, y

def get_test_data():
    print("Wczytywanie zbioru TESTOWEGO...")
    X = load_mnist_images('Minst/t10k-images-idx3-ubyte.gz')
    y = load_mnist_labels('Minst/t10k-labels-idx1-ubyte.gz')
    return X, y
