import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import os

# Import Twojej biblioteki
import utils
from network import MLP
from layers import Dense
import activations as act
import losses

# --- KONFIGURACJA EKSPERYMENTU ---
DATA_PATH = "mnist_data/"  # Upewnij się, że ścieżka jest poprawna

PARAM_GRID = {
    "learning_rate": [0.5,0.2,0.1, 0.05, 0.01],
    "hidden_layers": [[64], [128], [256, 128], [128, 64],[64,32]],
    "epochs": [5,10,15,20,30],
    "batch_size": [16,32,64,128],
}


# 1. Funkcja pomocnicza do liczenia dokładności
def calculate_accuracy(model, X, y_true):
    y_true_indices = np.argmax(y_true, axis=1)
    probs = model.predict(X)
    y_pred_indices = np.argmax(probs, axis=1)
    return np.mean(y_pred_indices == y_true_indices)


# 2. Wczytanie danych
def get_data():
    print("Wczytywanie danych MNIST...")
    img_path = os.path.join(DATA_PATH, "train-images-idx3-ubyte.gz")
    lbl_path = os.path.join(DATA_PATH, "train-labels-idx1-ubyte.gz")
    test_img_path = os.path.join(DATA_PATH, "t10k-images-idx3-ubyte.gz")
    test_lbl_path = os.path.join(DATA_PATH, "t10k-labels-idx1-ubyte.gz")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Nie znaleziono plików w folderze {DATA_PATH}")

    X_train = utils.load_mnist_images(img_path)
    y_train = utils.load_mnist_labels(lbl_path)
    X_test = utils.load_mnist_images(test_img_path)
    y_test = utils.load_mnist_labels(test_lbl_path)

    return X_train, y_train, X_test, y_test


# 3. Pojedynczy eksperyment (POPRAWIONY)
def run_single_experiment(
    X_train, y_train, X_test, y_test, lr, layers_struct, epochs, batch_size
):
    input_dim = 784

    model = MLP(learning_rate=lr)

    # Budowanie warstw
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
    loss_history = []

    # --- ZMIANA: Ręczna pętla treningowa ---
    # Dzięki temu liczymy loss niezależnie od parametru verbose w bibliotece
    for epoch in range(epochs):
        
        # Trenujemy 1 epokę
        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=False)

        # Ręcznie obliczamy stratę na podzbiorze danych (żeby było szybko)
        # Bierzemy losowe 2000 próbek do estymacji błędu
        indices = np.random.choice(len(X_train), 2000, replace=False)
        X_sample = X_train[indices]
        y_sample = y_train[indices]

        y_pred = model.predict(X_sample)
        loss = losses.cross_entropy_loss(y_sample, y_pred)
        loss_history.append(loss)

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
    print(
        f"{'ID':<4} | {'LR':<6} | {'Architektura':<20} |{'Epoki':<6} | {'Czas [s]':<8} | {'Accuracy %':<10}"
    )
    print("-" * 80)

    for i, params in enumerate(experiments):
        lr = params["learning_rate"]
        struct = params["hidden_layers"]
        epochs = params["epochs"]
        bs = params["batch_size"]

        acc, hist, duration = run_single_experiment(
            X_train, y_train, X_test, y_test, lr, struct, epochs, bs
        )

        res_entry = {
            "id": i,
            "lr": lr,
            "layers": str(struct),
            "epochs": epochs,     
            "batch_size": bs,
            "accuracy": acc,
            "duration": duration,
            "history": hist,
        }
        results.append(res_entry)

        print(
            f"{i:<4} | {lr:<6} | {str(struct):<20} | {epochs:<6} | {duration:<8.2f} | {acc * 100:.2f}%"
        )

    # --- RAPORT ---
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="accuracy", ascending=False)

    print("\n" + "=" * 30)
    print(" RANKING WYNIKÓW (MNIST)")
    print("=" * 30)
    df_display = df_sorted.copy()
    df_display["accuracy"] = df_display["accuracy"].apply(lambda x: f"{x * 100:.2f}%")
    print(
        df_display[["id", "lr", "layers","epochs","batch_size", "accuracy", "duration"]].to_string(
            index=False
        )
    )

    plot_mnist_results(df_sorted[:10], results)  


def plot_mnist_results(df, all_results):
    plt.style.use("bmh")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Wykres 1: Accuracy
    labels = [f"LR={r['lr']}\n{r['layers']}" for _, r in df.iterrows()]
    accs = df["accuracy"].values * 100

    axes[0].barh(range(len(accs)), accs, color="royalblue")
    axes[0].set_yticks(range(len(accs)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Accuracy [%] (Więcej = Lepiej)")
    axes[0].set_title("Dokładność modeli na zbiorze testowym")
    # Skalowanie osi X, żeby było widać różnice
    min_acc = max(0, min(accs) - 5)
    axes[0].set_xlim(min_acc, 100)

    # Wykres 2: Loss History
    axes[1].set_title("Spadek funkcji kosztu (Top 3 modele)")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Cross Entropy Loss")

    top_3 = df.iloc[:3]

    for _, row in top_3.iterrows():
        hist = next(r["history"] for r in all_results if r["id"] == row["id"])
        label = f"ID {row['id']}: {row['layers']} (Acc: {row['accuracy'] * 100:.1f}%)"
        axes[1].plot(hist, label=label, linewidth=2)

    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
