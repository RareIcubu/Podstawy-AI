import numpy as np
import matplotlib.pyplot as plt
def read_bin_raw(filename):
    try:
        with open(filename, 'rb') as f:  
            data = f.read()
            obrazy = []
            for i in range(0,len(data),784):  
                obrazy.append(data[i:i+784])
            return obrazy
    except FileNotFoundError:
        print(f"Błąd: Plik '{filename}' nie został znaleziony.")
    except Exception as e:
        print(f"Wystąpił błąd podczas odczytu pliku: {e}")

baza = read_bin_raw("./Minst/minst0.bin")
if baza is not None:
    idx = 0 
    
    plt.figure(figsize=(4, 4))
    plt.imshow(np.frombuffer(baza[idx], dtype=np.uint8).reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.show()