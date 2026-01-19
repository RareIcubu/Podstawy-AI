import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import os

# Import Twojej biblioteki
import utils # Twój plik do wczytywania MNIST
from network import MLP
from layers import Dense
import activations as act
import losses

# --- KONFIGURACJA EKSPERYMENTU ---

# Ścieżka do folderu z danymi (dostosuj jeśli masz inną nazwę)
DATA_PATH = 'mnist_data/' 

PARAM_GRID = {
    'learning_rate': [0.1, 0.05, 0.01],
    'hidden_layers': [
        [64],               # Mała sieć (szybka)
        [128],              # Standardowa
        [128, 64],          # Dwuwarstwowa (zwężająca się)
        [256, 128]          # Duża sieć
    ],
    'epochs': [5],          # MNIST uczy się szybko, 5-10 epok wystarczy do porównania
    'batch_size': [64]
}

# 1. Funkcja pomocnicza do liczenia dokładności
def calculate_accuracy(model, X, y_true):
    # y_true jest one-hot encoded, musimy zamienić na indeksy
    y_true_indices = np.argmax(y_true, axis=1)
    
    # Predykcja
    probs = model.predict(X)
    y_pred_indices = np.argmax(probs, axis=1)
    
    return np.mean(y_pred_indices == y_true_indices)

# 2. Wczytanie danych (tylko raz, przed pętlą)
def get_data():
    print("Wczytywanie danych MNIST...")
    # Sprawdź nazwy plików - muszą pasować do tych w Twoim folderze
    img_path = os.path.join(DATA_PATH, 'train-images-idx3-ubyte.gz')
    lbl_path = os.path.join(DATA_PATH, 'train-labels-idx1-ubyte.gz')
    
    # Używamy mniejszego zbioru walidacyjnego (np. testowego) do szybkiej oceny
    test_img_path = os.path.join(DATA_PATH, 't10k-images-idx3-ubyte.gz')
    test_lbl_path = os.path.join(DATA_PATH, 't10k-labels-idx1-ubyte.gz')
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Nie znaleziono plików w folderze {DATA_PATH}")

    X_train = utils.load_mnist_images(img_path)
    y_train = utils.load_mnist_labels(lbl_path)
    X_test = utils.load_mnist_images(test_img_path)
    y_test = utils.load_mnist_labels(test_lbl_path)
    
    return X_train, y_train, X_test, y_test

# 3. Pojedynczy eksperyment
def run_single_experiment(X_train, y_train, X_test, y_test, lr, layers_struct, epochs, batch_size):
    input_dim = 784 # Stała dla MNIST
    
    model = MLP(learning_rate=lr)
    
    # Budowanie warstw ukrytych (ReLU jest standardem dla obrazów)
    first = True
    for size in layers_struct:
        if first:
            model.add(Dense(size, act.relu, act.relu_prime, input_dim=input_dim))
            first = False
        else:
            model.add(Dense(size, act.relu, act.relu_prime))
            
    # Warstwa wyjściowa (Softmax dla klasyfikacji 10 cyfr)
    model.add(Dense(10, act.softmax, act.softmax_prime_dummy))
    
    model.compile(losses.cross_entropy_loss, losses.cross_entropy_prime)
    
    start_time = time.time()
    
    # Trening
    # Uwaga: fit zwraca historię LOSS, a nie Accuracy
    loss_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=False)
            
    duration = time.time() - start_time
    
    # Ewaluacja końcowa na zbiorze testowym
    final_acc = calculate_accuracy(model, X_test, y_test)
    
    return final_acc, loss_history, duration

# --- GŁÓWNA PĘTLA ---

def main():
    try:
        X_train, y_train, X_test, y_test = get_data()
    except Exception as e:
        print(f"Błąd wczytywania danych: {e}")
        return

    results = []
    
    keys, values = zip(*PARAM_GRID.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nZnaleziono {len(experiments)} kombinacji do przetestowania.")
    print("-" * 80)
    print(f"{'ID':<4} | {'LR':<6} | {'Architektura':<20} | {'Czas [s]':<8} | {'Accuracy %':<10}")
    print("-" * 80)
    
    for i, params in enumerate(experiments):
        lr = params['learning_rate']
        struct = params['hidden_layers']
        epochs = params['epochs']
        bs = params['batch_size']
        
        acc, hist, duration = run_single_experiment(
            X_train, y_train, X_test, y_test, lr, struct, epochs, bs
        )
        
        res_entry = {
            'id': i,
            'lr': lr,
            'layers': str(struct),
            'accuracy': acc,
            'duration': duration,
            'history': hist
        }
        results.append(res_entry)
        
        print(f"{i:<4} | {lr:<6} | {str(struct):<20} | {duration:<8.2f} | {acc*100:.2f}%")

    # --- RAPORT ---
    df = pd.DataFrame(results)
    
    # Sortujemy malejąco po Accuracy (najlepsze na górze)
    df_sorted = df.sort_values(by='accuracy', ascending=False)
    
    print("\n" + "="*30)
    print(" RANKING WYNIKÓW (MNIST)")
    print("="*30)
    # Wyświetlamy jako procenty dla czytelności
    df_display = df_sorted.copy()
    df_display['accuracy'] = df_display['accuracy'].apply(lambda x: f"{x*100:.2f}%")
    print(df_display[['id', 'lr', 'layers', 'accuracy', 'duration']].to_string(index=False))

    plot_mnist_results(df_sorted, results)

def plot_mnist_results(df, all_results):
    plt.style.use('bmh')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Wykres 1: Accuracy (Słupkowy)
    labels = [f"LR={r['lr']}\n{r['layers']}" for _, r in df.iterrows()]
    accs = df['accuracy'].values * 100 # Konwersja na %
    
    axes[0].barh(range(len(accs)), accs, color='royalblue')
    axes[0].set_yticks(range(len(accs)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Accuracy [%] (Więcej = Lepiej)")
    axes[0].set_title("Dokładność modeli na zbiorze testowym")
    # Ograniczenie osi X żeby było widać różnice (zwykle wszystko jest > 80%)
    axes[0].set_xlim(min(accs)-5, 100) 
    
    # Wykres 2: Loss History (Top 3)
    axes[1].set_title("Spadek funkcji kosztu (Top 3 modele)")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Cross Entropy Loss")
    
    top_3 = df.iloc[:3]
    
    for _, row in top_3.iterrows():
        hist = next(r['history'] for r in all_results if r['id'] == row['id'])
        label = f"ID {row['id']}: {row['layers']} (Acc: {row['accuracy']*100:.1f}%)"
        axes[1].plot(hist, label=label, linewidth=2)
        
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
