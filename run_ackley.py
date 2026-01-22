import numpy as np
import matplotlib.pyplot as plt
from network import MLP
from layers import Dense
import activations as act
import losses

# 1. Funkcja Ackleya
def ackley(x1, x2):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    return term1 + term2 + np.exp(1) + 20

# 2. Generowanie danych - WIĘCEJ I GĘŚCIEJ
print("Generowanie danych (20k próbek)...")
# Zwiększamy do 20 000 punktów. Przy funkcji wysokiej częstotliwości 
# musimy mieć gęste pokrycie, inaczej sieć zgaduje co jest między punktami.
X_raw = np.random.uniform(-2, 2, (20000, 2))

# --- FEATURE ENGINEERING ---
# Podajemy X, Y oraz ich COSINUSY (bo Ackley jest zbudowany z cosinusów).
# Sinusy można pominąć, cosinusy są tu kluczowe.
X_train = np.column_stack([
    X_raw[:, 0], 
    X_raw[:, 1], 
    np.cos(2 * np.pi * X_raw[:, 0]), 
    np.cos(2 * np.pi * X_raw[:, 1])
])

y_train_raw = ackley(X_raw[:, 0], X_raw[:, 1]).reshape(-1, 1)

# Skalowanie Y
y_max = np.max(y_train_raw)
y_train = y_train_raw / y_max 

# 3. Model
# Zaczynamy z dość wysokim LR, żeby szybko złapać kształt
model = MLP(learning_rate=0.05) 

# input_dim=4 (x, y, cos_x, cos_y)
model.add(Dense(128, act.tanh, act.tanh_prime, input_dim=4)) # Szersza warstwa
model.add(Dense(128, act.tanh, act.tanh_prime))
model.add(Dense(1, act.linear, act.linear_prime))

model.compile(losses.mse_loss, losses.mse_prime)

# 4. Trening z LR DECAY (Klucz do sukcesu)
print("Start treningu...")

EPOCHS = 500
history = []

for epoch in range(EPOCHS):
    # Dynamiczna zmiana Learning Rate
    if epoch == 500:
        model.learning_rate = 0.01
    if epoch == 1000:
        model.learning_rate = 0.005
    if epoch == 1500:
        model.learning_rate = 0.001

    # Wykonujemy trening (1 epoka) bez wypisywania logów
    model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=False)
    
    # --- POPRAWKA: Ręczne obliczenie straty ---
    # Ponieważ verbose=False, fit zwraca pustą listę. 
    # Musimy sami policzyć loss, żeby narysować wykres.
    
    # Robimy predykcję na całym zbiorze (szybkie dla NumPy)
    y_pred_full = model.predict(X_train)
    loss = losses.mse_loss(y_train, y_pred_full)
    history.append(loss)
    
    # Wypisujemy status co 100 epok
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss:.6f}")
# 5. Wizualizacja
print("Rysowanie i zapisywanie do Ackley.png...")
x = np.linspace(-2, 2, 100) # Mniejsza siatka do wizualizacji 3D dla czytelności
y = np.linspace(-2, 2, 100)
X_grid, Y_grid = np.meshgrid(x, y)

X_flat_raw = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

# Feature Engineering dla danych testowych/wykresu
X_flat_expanded = np.column_stack([
    X_flat_raw[:, 0], 
    X_flat_raw[:, 1], 
    np.cos(2 * np.pi * X_flat_raw[:, 0]), 
    np.cos(2 * np.pi * X_flat_raw[:, 1])
])

Z_true = ackley(X_grid, Y_grid)
Z_pred = model.predict(X_flat_expanded).reshape(X_grid.shape) * y_max 

# Wykresy
fig = plt.figure(figsize=(18, 6))

# 1. Oryginał 3D
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(X_grid, Y_grid, Z_true, cmap='viridis', alpha=0.8)
ax1.set_title("Oryginał (3D)")
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')

# 2. Aproksymacja 3D
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(X_grid, Y_grid, Z_pred, cmap='viridis', alpha=0.8)
ax2.set_title("Sieć MLP (3D)")
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')

# 3. Wykres błędu (Loss)
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(history)
ax3.set_title("Funkcja straty (MSE)")
ax3.set_xlabel("Epoka")
ax3.set_yscale('log')
ax3.grid(True)

plt.tight_layout()
plt.savefig('ackley_result.png')
print("Wykres zapisano do pliku ackley_result.png")
# plt.show()
