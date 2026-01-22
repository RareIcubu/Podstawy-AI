import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time

from network import MLP
from layers import Dense
import activations as act
import losses

# --- KONFIGURACJA EKSPERYMENTU (Ackley) ---
PARAM_GRID = {
    'learning_rate': [0.05, 0.01, 0.005],
    'hidden_layers': [
        [64, 64],
        [128, 128],
        [128, 128, 128],
        [256, 256]
    ],
    'epochs': [500, 1000],  # Dodane, aby sprawdzić wpływ długości treningu
    'batch_size': [64, 128] # Dodane
}

def get_data():
    print("Generowanie danych (Ackley)...")
    def generate_ackley_data(n_samples):
        X_raw = np.random.uniform(-2, 2, (n_samples, 2))
        X_feat = np.column_stack([
            X_raw[:, 0], 
            X_raw[:, 1], 
            np.cos(2 * np.pi * X_raw[:, 0]), 
            np.cos(2 * np.pi * X_raw[:, 1])
        ])
        
        def ackley(x1, x2):
            term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
            term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
            return term1 + term2 + np.exp(1) + 20

        y_raw = ackley(X_raw[:, 0], X_raw[:, 1]).reshape(-1, 1)
        return X_feat, y_raw

    X_train, y_train_raw = generate_ackley_data(10000)
    X_val, y_val_raw = generate_ackley_data(2500)
    
    y_max = np.max(y_train_raw)
    y_train = y_train_raw / y_max
    y_val = y_val_raw / y_max
    
    return X_train, y_train, X_val, y_val, y_max

def run_single_experiment(X_train, y_train, X_val, y_val, lr, layers_struct, epochs, batch_size):
    input_dim = X_train.shape[1] 
    model = MLP(learning_rate=lr)
    
    first = True
    for size in layers_struct:
        if first:
            model.add(Dense(size, act.tanh, act.tanh_prime, input_dim=input_dim))
            first = False
        else:
            model.add(Dense(size, act.tanh, act.tanh_prime))
            
    model.add(Dense(1, act.linear, act.linear_prime))
    model.compile(losses.mse_loss, losses.mse_prime)
    
    start_time = time.time()
    
    # Trening pętlą z walidacją
    history = []
    
    # Aby przyspieszyć, mierzymy loss co pewien czas, ale trenujemy normalnie batchami
    steps = epochs // 50 if epochs >= 50 else 1
    
    for i in range(epochs):
        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=False)
        
        # Log co 'steps' epok lub na końcu
        if i % steps == 0 or i == epochs - 1:
            preds = model.predict(X_val)
            loss = losses.mse_loss(y_val, preds)
            history.append(loss)
            
    duration = time.time() - start_time
    final_loss = history[-1]
    
    return final_loss, history, duration

def main():
    X_train, y_train, X_val, y_val, y_max = get_data()
    
    keys, values = zip(*PARAM_GRID.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nLiczba kombinacji: {len(experiments)}")
    
    # Logowanie do pliku
    with open("ackley_tuning_log.txt", "w") as log_file:
        header = f"{'ID':<4} | {'LR':<6} | {'Layers':<20} | {'Ep':<4} | {'BS':<4} | {'Time[s]':<7} | {'ValLoss':<8}"
        print("-" * 80)
        print(header)
        print("-" * 80)
        log_file.write(header + "\n")
        log_file.write("-" * 80 + "\n")

        results = []
        
        for i, p in enumerate(experiments):
            lr = p['learning_rate']
            struct = p['hidden_layers']
            ep = p['epochs']
            bs = p['batch_size']
            
            loss, hist, duration = run_single_experiment(X_train, y_train, X_val, y_val, lr, struct, ep, bs)
            
            res_entry = {
                'id': i,
                'lr': lr,
                'layers': str(struct),
                'epochs': ep,
                'batch_size': bs,
                'loss': loss,
                'duration': duration,
                'history': hist
            }
            results.append(res_entry)
            
            line = f"{i:<4} | {lr:<6} | {str(struct):<20} | {ep:<4} | {bs:<4} | {duration:<7.2f} | {loss:.6f}"
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='loss')
    
    print("\nTOP 5 WYNIKÓW (Najmniejszy Błąd):")
    print(df_sorted[['id', 'lr', 'layers', 'epochs', 'batch_size', 'loss']].head(5).to_string(index=False))

    plot_results(df_sorted)

def plot_results(df):
    plt.style.use('bmh')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Bar chart Loss
    top_20 = df.head(20) # Pokazujemy top 20 żeby było czytelnie
    labels = [f"ID {r['id']}\nLR={r['lr']} L={r['layers']}" for _, r in top_20.iterrows()]
    losses = top_20['loss'].values
    
    axes[0].barh(range(len(losses)), losses, color='teal')
    axes[0].set_yticks(range(len(losses)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Top 20 Konfiguracji (MSE Loss)")
    axes[0].set_xlabel("MSE (Mniej = Lepiej)")
    
    # 2. History
    top_3 = df.head(3)
    for _, row in top_3.iterrows():
        axes[1].plot(row['history'], label=f"ID {row['id']} (Loss: {row['loss']:.4f})")
        
    axes[1].set_title("Przebieg uczenia (Top 3)")
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, which="both", ls="-")
    
    plt.tight_layout()
    plt.savefig('ackley_tuning_results.png')
    print("\nZapisano wykres zbiorczy do: ackley_tuning_results.png")

if __name__ == "__main__":
    main()