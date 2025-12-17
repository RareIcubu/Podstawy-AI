import siec as sc
import warstwy as wr
import funkcje as fa
import numpy as np
import minst as mn
import time

# --- 1. WCZYTANIE OFICJALNYCH DANYCH ---
print("Wczytywanie danych...")
# To trwa sekunde
train_X, train_y = mn.get_training_data()
test_X, test_y = mn.get_test_data()

print(f"Mamy {len(train_X)} przykładów. To wystarczy na >95%.")

# --- 2. MODEL ---
def create_model(input_size, output_size=10):
    model = sc.MLP(learning_rate=0.1)
    
    # Architektura (Solidna)
    model.add_layer(wr.Warstwa(input_size=input_size, layer_size=256, activation=fa.relu, activation_prime=fa.relu_derivative, weights_init='xavier'))
    model.add_layer(wr.Warstwa(input_size=256, layer_size=128, activation=fa.relu, activation_prime=fa.relu_derivative, weights_init='xavier'))
    model.add_layer(wr.Warstwa(input_size=128, layer_size=output_size, activation=fa.softmax, activation_prime=fa.derivative_one, weights_init='xavier'))
    return model

model = create_model(784)

# --- 3. TRENING ---
# Zwiększamy Batch Size do 128 lub 256. 
# Im większy batch, tym mniej operacji Pythona, a więcej szybkiego NumPy.
BATCH_SIZE = 128 
EPOCHS = 5 

print(f"\nStart treningu (Batch: {BATCH_SIZE})...")
start_time = time.time()

for epoch in range(EPOCHS):
    # Prosty LR Decay
    if epoch == 8:  model.learning_rate = 0.05
    if epoch == 12: model.learning_rate = 0.01

    # Tasowanie (NumPy robi to błyskawicznie)
    perm = np.random.permutation(len(train_X))
    train_X = train_X[perm]
    train_y = train_y[perm]

    # Pętla treningowa
    for i in range(0, len(train_X), BATCH_SIZE):
        batch_X = train_X[i : i + BATCH_SIZE]
        batch_y = train_y[i : i + BATCH_SIZE]
        model.fit(batch_X, batch_y, epochs=1, print_every=0)

    # Ewaluacja
    acc = model.evaluate(test_X, test_y)
    elapsed = time.time() - start_time
    print(f"Epoka {epoch+1}/{EPOCHS} | Czas: {elapsed:.0f}s | Acc: {acc*100:.2f}%")

model.save_model('model_mnist_fast.npz')
print("\nGotowe. To nie powinno boleć.")
