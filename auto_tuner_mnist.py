import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import os

import utils
from network import MLP
from layers import Dense
import activations as act
import losses

# --- KONFIGURACJA EKSPERYMENTU ---
DATA_PATH = "mnist_data/"

# Rozszerzona siatka parametrów zgodnie z prośbą
PARAM_GRID = {
    "learning_rate": [0.1, 0.05, 0.01],
    "hidden_layers": [[64], [128], [128, 64], [256, 128]],
    "epochs": [5, 10],      # Dodane
    "batch_size": [32, 64]  # Dodane
}

def calculate_accuracy(model, X, y_true):
    y_true_indices = np.argmax(y_true, axis=1)
    probs = model.predict(X)
    y_pred_indices = np.argmax(probs, axis=1)
    return np.mean(y_pred_indices == y_true_indices)

def get_data():
    print("Wczytywanie danych MNIST...")
    # Ścieżki do plików .gz (muszą być w folderze mnist_data/)
    paths = {
        'train_img': os.path.join(DATA_PATH, "train-images-idx3-ubyte.gz"),
        'train_lbl': os.path.join(DATA_PATH, "train-labels-idx1-ubyte.gz"),
        'test_img':  os.path.join(DATA_PATH, "t10k-images-idx3-ubyte.gz"),
        'test_lbl':  os.path.join(DATA_PATH, "t10k-labels-idx1-ubyte.gz")
    }

    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Brak pliku: {p}")

    X_full = utils.load_mnist_images(paths['train_img'])
    y_full = utils.load_mnist_labels(paths['train_lbl'])
    X_test = utils.load_mnist_images(paths['test_img'])
    y_test = utils.load_mnist_labels(paths['test_lbl'])
    
    # Walidacja (10k)
    val_size = 10000
    X_train = X_full[:-val_size]
    y_train = y_full[:-val_size]
    X_val = X_full[-val_size:]
    y_val = y_full[-val_size:]

    print(f"Dane: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_single_experiment(X_train, y_train, X_val, y_val, lr, layers_struct, epochs, batch_size):
    input_dim = 784
    model = MLP(learning_rate=lr)

    first = True
    for size in layers_struct:
        if first:
            model.add(Dense(size, act.relu, act.relu_prime, input_dim=input_dim))
            first = False
        else:
            model.add(Dense(size, act.relu, act.relu_prime))

    model.add(Dense(10, act.softmax, act.softmax_prime_dummy))
    model.compile(losses.cross_entropy_loss, losses.cross_entropy_prime)

    start_time = time.time()
    
    # Wykorzystanie parametrow epochs i batch_size
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_val, y_val),
        verbose=False
    )
    
    duration = time.time() - start_time
    val_acc = calculate_accuracy(model, X_val, y_val)
    
    return val_acc, history['loss'], duration

def main():
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    except Exception as e:
        print(f"Błąd: {e}")
        return

    keys, values = zip(*PARAM_GRID.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\nLiczba kombinacji: {len(experiments)}")
    
    # Otwieramy plik do logowania wyników
    with open("mnist_tuning_log.txt", "w") as log_file:
        header = f"{'ID':<4} | {'LR':<6} | {'Layers':<20} | {'Ep':<3} | {'BS':<4} | {'Time[s]':<7} | {'ValAcc%':<8}"
        print("-" * 80)
        print(header)
        print("-" * 80)
        log_file.write(header + "\n")
        log_file.write("-" * 80 + "\n")

        results = []

        for i, p in enumerate(experiments):
            lr = p["learning_rate"]
            struct = p["hidden_layers"]
            ep = p["epochs"]
            bs = p["batch_size"]

            acc, hist, duration = run_single_experiment(
                X_train, y_train, X_val, y_val, lr, struct, ep, bs
            )

            res_entry = {
                "id": i,
                "lr": lr,
                "layers": str(struct),
                "epochs": ep,
                "batch_size": bs,
                "accuracy": acc,
                "duration": duration,
                "history": hist
            }
            results.append(res_entry)

            # Log do konsoli
            line = f"{i:<4} | {lr:<6} | {str(struct):<20} | {ep:<3} | {bs:<4} | {duration:<7.2f} | {acc * 100:.2f}%"
            print(line)
            
            # Log do pliku (flush=True dla bezpieczeństwa)
            log_file.write(line + "\n")
            log_file.flush()

    # Sortowanie i wykresy
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="accuracy", ascending=False)

    print("\nTOP 5 WYNIKÓW:")
    print(df_sorted[["id", "lr", "layers", "epochs", "batch_size", "accuracy"]].head(5).to_string(index=False))

    plot_mnist_results(df_sorted)

def plot_mnist_results(df):
    plt.style.use("bmh")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Bar chart (Top 10)
    top_10 = df.head(10)
    labels = [f"ID {r['id']}\nLR={r['lr']} L={r['layers']}" for _, r in top_10.iterrows()]
    accs = top_10["accuracy"].values * 100

    axes[0].barh(range(len(accs)), accs, color="royalblue")
    axes[0].set_yticks(range(len(accs)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Top 10 Konfiguracji (Accuracy)")
    axes[0].set_xlabel("%")
    
    # Skalowanie osi X "zoom"
    if len(accs) > 0:
        axes[0].set_xlim(min(accs)-1, 100)

    # 2. Loss History (Top 3)
    top_3 = df.head(3)
    for _, row in top_3.iterrows():
        axes[1].plot(row["history"], label=f"ID {row['id']} (Acc: {row['accuracy']*100:.1f}%)")
    
    axes[1].set_title("Loss History (Top 3)")
    axes[1].set_yscale('log')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("mnist_tuning_results.png")
    print("\nZapisano wykres zbiorczy do: mnist_tuning_results.png")

if __name__ == "__main__":
    main()