import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time

# Import Twojej biblioteki
from network import MLP
from layers import Dense
import activations as act
import losses

# --- KONFIGURACJA EKSPERYMENTU ---

# 1. Jakie parametry chcemy przetestować?
PARAM_GRID = {
    'learning_rate': [0.05, 0.01, 0.005],
    'hidden_layers': [
        [64, 64],           # Mniejsza sieć
        [128, 128],         # Średnia sieć (Twój obecny standard)
        [128, 128, 128],    # Głębsza sieć
        [256, 256]          # Szersza sieć
    ],
    'epochs': [1000]        # Stała liczba epok dla porównania
}

# 2. Przygotowanie danych (Ackley + Feature Engineering)
def get_data():
    print("Generowanie danych (Ackley)...")
    # 10k próbek do szybkiego testu (do finalnego tuningu można dać 20k)
    X_raw = np.random.uniform(-2, 2, (10000, 2))
    
    # Feature Engineering (x, y, cos(x), cos(y)) - to co działało najlepiej
    X_train = np.column_stack([
        X_raw[:, 0], 
        X_raw[:, 1], 
        np.cos(2 * np.pi * X_raw[:, 0]), 
        np.cos(2 * np.pi * X_raw[:, 1])
    ])
    
    def ackley(x1, x2):
        term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
        term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
        return term1 + term2 + np.exp(1) + 20

    y_train_raw = ackley(X_raw[:, 0], X_raw[:, 1]).reshape(-1, 1)
    
    # Normalizacja Y
    y_max = np.max(y_train_raw)
    y_train = y_train_raw / y_max
    
    return X_train, y_train, y_max

# 3. Funkcja uruchamiająca pojedynczy trening
def run_single_experiment(X, y, lr, layers_struct, epochs):
    input_dim = X.shape[1] # 4
    
    model = MLP(learning_rate=lr)
    
    # Budowanie warstw ukrytych
    first = True
    for size in layers_struct:
        if first:
            model.add(Dense(size, act.tanh, act.tanh_prime, input_dim=input_dim))
            first = False
        else:
            model.add(Dense(size, act.tanh, act.tanh_prime))
            
    # Warstwa wyjściowa (Linear dla regresji)
    model.add(Dense(1, act.linear, act.linear_prime))
    
    model.compile(losses.mse_loss, losses.mse_prime)
    
    # Trening (bez verbose, żeby nie śmiecić)
    start_time = time.time()
    
    # Używamy pętli "ręcznej" żeby mieć pewność co do historii loss
    history = []
    # Prosta wersja bez LR decay dla równego porównania parametrów startowych
    # (można dodać decay, ale utrudnia to ocenę, który początkowy LR jest lepszy)
    for i in range(epochs):
        model.fit(X, y, epochs=1, batch_size=128, verbose=False)
        # Próbkowanie loss co 50 epok dla szybkości
        if i % 50 == 0 or i == epochs - 1:
            preds = model.predict(X)
            loss = losses.mse_loss(y, preds)
            history.append(loss)
            
    duration = time.time() - start_time
    final_loss = history[-1]
    
    return final_loss, history, duration

# --- GŁÓWNA PĘTLA ---

def main():
    X, y, y_max = get_data()
    results = []
    
    # Tworzymy wszystkie kombinacje parametrów
    keys, values = zip(*PARAM_GRID.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nZnaleziono {len(experiments)} kombinacji do przetestowania.")
    print("-" * 60)
    print(f"{'ID':<4} | {'LR':<8} | {'Architektura':<20} | {'Czas [s]':<8} | {'Loss':<10}")
    print("-" * 60)
    
    best_models_history = {} # Do wykresu
    
    for i, params in enumerate(experiments):
        lr = params['learning_rate']
        struct = params['hidden_layers']
        epochs = params['epochs']
        
        loss, hist, duration = run_single_experiment(X, y, lr, struct, epochs)
        
        # Zapisz wynik
        res_entry = {
            'id': i,
            'lr': lr,
            'layers': str(struct),
            'loss': loss,
            'duration': duration,
            'history': hist # przechowujemy historię do wyrysowania
        }
        results.append(res_entry)
        
        print(f"{i:<4} | {lr:<8} | {str(struct):<20} | {duration:<8.2f} | {loss:.6f}")

    # --- ANALIZA WYNIKÓW ---
    df = pd.DataFrame(results)
    
    # Sortowanie od najmniejszego błędu
    df_sorted = df.sort_values(by='loss')
    
    print("\n" + "="*30)
    print(" RANKING NAJLEPSZYCH MODELI")
    print("="*30)
    print(df_sorted[['id', 'lr', 'layers', 'loss', 'duration']].to_string(index=False))
    
    best_id = df_sorted.iloc[0]['id']
    print(f"\nNajlepszy model: ID {best_id} (Loss: {df_sorted.iloc[0]['loss']:.6f})")

    # --- WYKRESY ---
    plot_results(df_sorted, results)

def plot_results(df, all_results):
    plt.style.use('bmh') # Ładny styl wykresów
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Wykres 1: Porównanie Loss dla wszystkich eksperymentów (Bar Chart)
    # Tworzymy etykiety opisowe
    labels = [f"LR={r['lr']}\n{r['layers']}" for _, r in df.iterrows()]
    losses = df['loss'].values
    
    axes[0].barh(range(len(losses)), losses, color='teal')
    axes[0].set_yticks(range(len(losses)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis() # Najlepsze na górze
    axes[0].set_xlabel("Final MSE Loss (Mniej = Lepiej)")
    axes[0].set_title("Ranking Konfiguracji")
    
    # Wykres 2: Krzywe uczenia dla TOP 3 modeli
    axes[1].set_title("Przebieg uczenia (Top 3 modele)")
    axes[1].set_xlabel("Krok pomiarowy (co 50 epok)")
    axes[1].set_ylabel("Loss (log scale)")
    
    top_3 = df.iloc[:3]
    
    for _, row in top_3.iterrows():
        # Znajdź historię w oryginalnej liście results po ID
        hist = next(r['history'] for r in all_results if r['id'] == row['id'])
        label = f"ID {row['id']}: LR={row['lr']}, {row['layers']}"
        axes[1].plot(hist, label=label, linewidth=2)
        
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, which="both", ls="-")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
