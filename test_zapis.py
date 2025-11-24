import numpy as np
import os

# Import Twoich klas
from siec import MLP
from warstwy import WarstwaUkryta
import funkcje as fa  # Zakładamy, że plik nazywa się funkcje.py

def main():
    print("--- TEST ZAPISU I ODCZYTU MODELU ---")
    
    # 1. Dane XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]) # One-hot

    # 2. Tworzenie i szybki trening modelu
    print("\n1. Trenowanie modelu (krótki trening)...")
    model_original = MLP(learning_rate=0.1)
    model_original.add_layer(WarstwaUkryta(2, 4, fa.relu, fa.relu_derivative))
    model_original.add_layer(WarstwaUkryta(4, 2, fa.softmax, fa.derivative_one))
    
    # Trenujemy krótko, nie zależy nam na idealnej dokładności, 
    # tylko na tym, żeby wagi się zmieniły od wartości losowych.
    model_original.fit(X, y, epochs=10000)

    # 3. Pobranie predykcji z ORYGINALNEGO modelu
    print("   Pobieranie predykcji z modelu w pamięci...")
    pred_original = model_original.predict_proba(X)
    print(f"   Przykładowa wartość (pierwsza próbka): {pred_original[0]}")

    # 4. Zapis modelu
    filename = "./test_model_xor.npz"
    print(f"\n2. Zapisywanie modelu do pliku: {filename}")
    model_original.save_model(filename)
    
    # Sprawdzenie czy plik powstał
    if os.path.exists(filename):
        print("   Plik został utworzony poprawnie.")
    else:
        print("   BŁĄD: Nie znaleziono pliku!")
        return

    # 5. Wczytanie modelu do NOWEJ zmiennej
    print("\n3. Wczytywanie modelu z pliku do nowej zmiennej...")
    try:
        model_loaded = MLP.load_model(filename)
    except Exception as e:
        print(f"   BŁĄD podczas wczytywania: {e}")
        return

    # 6. Porównanie wyników
    print("\n4. Porównanie wyników (Oryginał vs Wczytany)...")
    pred_loaded = model_loaded.predict_proba(X)
    
    # Sprawdzamy czy tablice są identyczne (co do bardzo małego marginesu błędu float)
    are_equal = np.allclose(pred_original, pred_loaded)

    if are_equal:
        print("\nSUKCES: Wyniki są identyczne!")
        print("Mechanizm zapisu i odczytu działa poprawnie.")
        print("-" * 30)
        print(f"Oryginał [0]: {pred_original[0]}")
        print(f"Wczytany [0]: {pred_loaded[0]}")
    else:
        print("\nBŁĄD: Wyniki się różnią!")
        print(f"Oryginał [0]: {pred_original[0]}")
        print(f"Wczytany [0]: {pred_loaded[0]}")

    # Sprzątanie (usunięcie pliku testowego)
    # Odkomentuj poniższą linię, jeśli chcesz, żeby program kasował plik po teście
    # os.remove(filename)

if __name__ == "__main__":
    main()