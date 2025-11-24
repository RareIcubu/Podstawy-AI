import numpy as np
from siec import MLP
from warstwy import WarstwaUkryta
import funkcje as fa

def main():
    print("--- TEST XOR (Bez zewnętrznych bibliotek) ---")

    # 1. Przygotowanie danych (Bramka XOR)
    # Wejście: [A, B]
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Wyjście oczekiwane (One-Hot Encoding)
    # Twoja sieć używa Softmax i CrossEntropy, więc wyjście musi być wektorem prawdopodobieństw.
    # Format: [Prawdopodobieństwo_Zera, Prawdopodobieństwo_Jedynki]
    # 0 XOR 0 = 0 -> [1, 0]
    # 0 XOR 1 = 1 -> [0, 1]
    # 1 XOR 0 = 1 -> [0, 1]
    # 1 XOR 1 = 0 -> [1, 0]
    y = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ])

    # 2. Budowa Sieci
    # Learning rate może być wyższy przy tak prostym problemie
    model = MLP(learning_rate=0.1)

    # Warstwa ukryta: 2 wejścia -> 4 neurony (wystarczą, by uchwycić nieliniowość)
    model.add_layer(WarstwaUkryta(
        input_size=2, 
        layer_size=4, 
        activation=fa.relu, 
        activation_prime=fa.relu_derivative,
        weights_init='he'
    ))

    # Warstwa wyjściowa: 4 neurony -> 2 klasy (0 lub 1)
    model.add_layer(WarstwaUkryta(
        input_size=4, 
        layer_size=2, 
        activation=fa.softmax, 
        activation_prime=fa.derivative_one,
        weights_init='xavier'
    ))

    # 3. Trening
    # XOR bywa "wredny" dla sieci neuronowych, potrzebuje czasem sporo epok, żeby "zaskoczyć".
    print("Rozpoczynam trening...")
    model.fit(X, y, epochs=5000)

    # 4. Sprawdzenie wyników
    print("\n--- WYNIKI ---")
    print(f"{'Wejście':<15} | {'Oczekiwane':<10} | {'Przewidziane':<10} | {'Pewność sieci':<20}")
    print("-" * 65)

    predictions = model.predict_proba(X)
    
    for i in range(len(X)):
        input_str = str(X[i])
        
        # Prawdziwa klasa (gdzie jest 1 w one-hot)
        true_class = np.argmax(y[i])
        
        # Przewidziana klasa (gdzie jest największe prawdopodobieństwo)
        pred_class = np.argmax(predictions[i])
        
        # Pewność (prawdopodobieństwo wybranej klasy)
        confidence = predictions[i][pred_class]
        
        status = "OK" if true_class == pred_class else "BŁĄD"
        
        print(f"{input_str:<15} | {true_class:<10} | {pred_class:<10} | {confidence:.4f} ({status})")

    # Prosta walidacja końcowa
    acc = model.evaluate(X, y)
    if acc == 1.0:
        print("\nSUKCES: Sieć poprawnie nauczyła się funkcji XOR!")
    else:
        print("\nPORAŻKA: Sieć nie zbiegła do rozwiązania. Spróbuj uruchomić ponownie.")

main()