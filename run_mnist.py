# plik: run_mnist.py
import numpy as np
import utils
from network import MLP
from layers import Dense
import activations as act
import losses

# 1. Wczytaj dane (pamiętaj o pobraniu plików .gz!)
print("Wczytywanie danych MNIST...")
# Zakładam, że pliki są w folderze 'mnist_data'
X_train = utils.load_mnist_images('mnist_data/train-images-idx3-ubyte.gz')
y_train = utils.load_mnist_labels('mnist_data/train-labels-idx1-ubyte.gz')
X_test = utils.load_mnist_images('mnist_data/t10k-images-idx3-ubyte.gz')
y_test = utils.load_mnist_labels('mnist_data/t10k-labels-idx1-ubyte.gz')

# 2. Konfiguracja sieci
net = MLP(learning_rate=0.1)

net.add(Dense(units=128, activation=act.relu, activation_prime=act.relu_prime, input_dim=784))

net.add(Dense(units=64, activation=act.relu, activation_prime=act.relu_prime))

net.add(Dense(units=10, activation=act.softmax, activation_prime=act.softmax_prime_dummy))

net.compile(losses.cross_entropy_loss, losses.cross_entropy_prime)
# 3. Trening
print("Start treningu...")
net.fit(X_train, y_train, epochs=5, batch_size=64)

# 4. Ewaluacja
preds = net.predict(X_test)
accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1))
print(f"Dokładność na zbiorze testowym: {accuracy * 100:.2f}%")

# 5. Zapis
net.save("mnist_model.pkl")
