import numpy as np
import matplotlib.pyplot as plt
import siec as sc
import warstwy as wr
import funkcje as fa

def read_minst(filename):
    with open(filename, 'rb') as f:  
        data = f.read()
        obrazy = []
        for i in range(0,len(data),784):  
            obrazy.append(np.frombuffer(data[i:i+784], dtype=np.uint8).astype(np.float32) / 255.0)
        return obrazy

def plot_minst(obraz):
    plt.figure(figsize=(5,5))
    plt.imshow(np.frombuffer(obraz, dtype=np.uint8).reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()
    
def create_dataset(rozmiar = 100,start = 0):
    obrazy = []
    for cyfra in range(10):
        filename = f'./Minst/minst{cyfra}.bin' 
        odczytany_minst = read_minst(filename)
        for nr_obrazu in range(start, start + rozmiar):
            obrazy.append([cyfra,odczytany_minst[nr_obrazu]])
    return obrazy
def see_dataset(rozmiar_jednej_bazy,obrazy):
    plt.figure(figsize=(10,10))
    for i in range(0,9*rozmiar_jednej_bazy,rozmiar_jednej_bazy):
        for j in range(10):
            plt.subplot(10,10,int(i/rozmiar_jednej_bazy)*10+ j+1)
            plt.axis('off')
            plt.imshow(np.frombuffer(obrazy[i+j][1], dtype=np.uint8).reshape(28, 28), cmap='gray')
    plt.show()
def mix_dataset(obrazy):
    np.random.shuffle(obrazy)
    return obrazy
def save_dataset(obrazy,filename):
    np.savez(filename, dataset=obrazy)
def load_dataset(filename):
    data = np.load(filename, allow_pickle=True)
    return data['dataset']


