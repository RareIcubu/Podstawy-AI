# plik: run_mnist.py
import numpy as np
import matplotlib.pyplot as plt
import utils
from network import MLP
from layers import Dense
import activations as act
import losses

# 1. Wczytaj dane (pamiętaj o pobraniu plików .gz!)
print("Wczytywanie danych MNIST...")
# Zakładam, że pliki są w folderze 'mnist_data'
X_train_full = utils.load_mnist_images('mnist_data/train-images-idx3-ubyte.gz')
y_train_full = utils.load_mnist_labels('mnist_data/train-labels-idx1-ubyte.gz')
X_test = utils.load_mnist_images('mnist_data/t10k-images-idx3-ubyte.gz')
y_test = utils.load_mnist_labels('mnist_data/t10k-labels-idx1-ubyte.gz')

# Podział na train/val
val_size = 10000
X_train = X_train_full[:-val_size]
y_train = y_train_full[:-val_size]
X_val = X_train_full[-val_size:]
y_val = y_train_full[-val_size:]
print(f"Dane treningowe: {len(X_train)}, Walidacyjne: {len(X_val)}, Testowe: {len(X_test)}")

# 2. Konfiguracja sieci
net = MLP(learning_rate=0.1)

net.add(Dense(units=128, activation=act.relu, activation_prime=act.relu_prime, input_dim=784))

net.add(Dense(units=64, activation=act.relu, activation_prime=act.relu_prime))

net.add(Dense(units=10, activation=act.softmax, activation_prime=act.softmax_prime_dummy))

net.compile(losses.cross_entropy_loss, losses.cross_entropy_prime)
# 3. Trening
print("Start treningu...")
# Zbieramy historię (fit zwraca słownik {'loss': [], 'val_loss': []})
history = net.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), patience=3)

# 4. Ewaluacja
print("Ewaluacja i generowanie wykresów...")
preds = net.predict(X_test)
y_pred_cls = np.argmax(preds, axis=1)
y_true_cls = np.argmax(y_test, axis=1)

accuracy = np.mean(y_pred_cls == y_true_cls)
print(f"Dokładność na zbiorze testowym: {accuracy * 100:.2f}%")

# Macierz pomyłek (Ręczna implementacja, bez sklearn)
num_classes = 10
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true_cls, y_pred_cls):
    confusion_matrix[t, p] += 1

# 5. Wizualizacja i Zapis
fig = plt.figure(figsize=(14, 6))

# Wykres Loss
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(history['loss'], label='Train Loss', marker='o')
if history['val_loss']:
    ax1.plot(history['val_loss'], label='Val Loss', marker='o')
ax1.set_title('Historia Uczenia (Loss)')
ax1.set_xlabel('Epoka')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Wykres Confusion Matrix
ax2 = fig.add_subplot(1, 2, 2)
im = ax2.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
ax2.set_title(f'Macierz Pomyłek (Accuracy: {accuracy*100:.2f}%)')
fig.colorbar(im, ax=ax2)

# Etykiety osi
tick_marks = np.arange(num_classes)
ax2.set_xticks(tick_marks)
ax2.set_yticks(tick_marks)
ax2.set_xlabel('Przewidziana klasa')
ax2.set_ylabel('Prawdziwa klasa')

# Wypisanie liczb w komórkach
thresh = confusion_matrix.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        ax2.text(j, i, format(confusion_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("mnist_results.png")
print("Zapisano wykresy do mnist_results.png")

# 6. Zapis modelu
net.save("mnist_model.pkl")
