import numpy as np

class MLP:
    """
    Implementacja wielowarstwowego perceptronu (MLP) w NumPy
    na potrzeby klasyfikacji.
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, weight_init='he', 
                 activation='relu', output_activation='softmax'):
        """
        Konstruktor sieci neuronowej.
        
        Parametry:
        - layer_sizes (list): Lista liczb całkowitych, np. [wejście, ukryta1, ukryta2, wyjście]
        - learning_rate (float): Współczynnik uczenia
        - weight_init (str): Metoda inicjalizacji wag ('he', 'xavier', 'random')
        - activation (str): Funkcja aktywacji dla warstw ukrytych ('relu', 'sigmoid')
        - output_activation (str): Funkcja aktywacji dla warstwy wyjściowej ('softmax', 'linear')
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.weights = []
        self.biases = []
        
        self._activation_funcs = {
            'relu': (self._relu, self._relu_derivative),
            'sigmoid': (self._sigmoid, self._sigmoid_derivative)
        }
        self._output_funcs = {
            'softmax': self._softmax,
            'linear': self._linear  # Dla regresji
        }

        # Wybór funkcji dla warstw ukrytych
        self.activation_name = activation
        self.activation_func, self.activation_derivative = self._activation_funcs[activation]
        
        # Wybór funkcji dla warstwy wyjściowej
        self.output_activation_name = output_activation
        self.output_func = self._output_funcs[output_activation]
        
        # Inicjalizacja wag i biasów
        self._initialize_weights()

        # Zmienne do przechowywania stanów pośrednich (potrzebne do backpropagation)
        self.z = []  # Przechowuje wejścia liniowe (przed aktywacją)
        self.a = []  # Przechowuje wyjścia (po aktywacji)

    # --- Inicjalizacja Wag ---
    
    def _initialize_weights(self):
        """Inicjalizuje wagi i biasy dla wszystkich warstw."""
        # Pętla od 0 do (liczba_warstw - 2)
        # Sieć o N warstwach (np. [784, 128, 10]) ma N-1 zestawów wag/biasów
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            # Wybór metody inicjalizacji
            if self.weight_init == 'xavier':
                # Inicjalizacja Xavier/Glorot: dobra dla sigmoid
                std_dev = np.sqrt(1 / input_size)
            elif self.weight_init == 'he':
                # Inicjalizacja He: dobra dla ReLU
                std_dev = np.sqrt(2 / input_size)
            else:
                # Prosta losowa inicjalizacja (zwykle gorsza)
                std_dev = 0.01
                
            W = np.random.randn(input_size, output_size) * std_dev
            b = np.zeros((1, output_size))
            
            self.weights.append(W)
            self.biases.append(b)

    # --- Funkcje Aktywacji i Ich Pochodne ---

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        # Zwraca 1.0 tam, gdzie z > 0, i 0.0 w przeciwnym razie
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        sig = self._sigmoid(z)
        return sig * (1 - sig)

    def _linear(self, z):
        # Aktywacja liniowa (brak aktywacji)
        return z

    def _softmax(self, z):
        # Stabilna numerycznie implementacja softmax
        # Przesunięcie o max(z) zapobiega dużym liczbom w np.exp()
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # --- Propagacja do Przodu (Forward Propagation) ---

    def _forward(self, X):
        """Wykonuje propagację do przodu."""
        self.a = [X]  # Zerowa aktywacja to dane wejściowe
        self.z = []   # Lista na wejścia liniowe (przed aktywacją)
        
        num_layers = len(self.weights)
        
        # Pętla przez wszystkie warstwy (Wagi i Biasy)
        for i in range(num_layers):
            W = self.weights[i]
            b = self.biases[i]
            
            # Poprzednia aktywacja (lub X dla pierwszej warstwy)
            a_prev = self.a[-1] 
            
            # Krok liniowy: z = a_prev * W + b
            z_curr = np.dot(a_prev, W) + b
            self.z.append(z_curr)
            
            # Krok aktywacji
            if i == num_layers - 1:
                # Ostatnia warstwa: użyj funkcji wyjściowej (np. softmax)
                a_curr = self.output_func(z_curr)
            else:
                # Warstwy ukryte: użyj funkcji aktywacji (np. relu)
                a_curr = self.activation_func(z_curr)
                
            self.a.append(a_curr)
            
        return self.a[-1]

    # --- Funkcja Straty ---

    def _compute_cross_entropy_loss(self, y_true, y_pred):
        """Oblicza stratę entropii krzyżowej."""
        m = y_true.shape[0]  # Liczba próbek
        
        # Przycięcie wartości, aby uniknąć log(0) (stabilność numeryczna)
        y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
        
        # Wzór na entropię krzyżową: -sum(y_true * log(y_pred))
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        return loss

    # --- Propagacja Wsteczna (Backward Propagation) ---

    def _backward(self, X, y_true):
        """Wykonuje propagację wsteczną i aktualizuje wagi."""
        m = y_true.shape[0]  # Liczba próbek
        y_pred = self.a[-1]  # Wyjście z forward pass

        # --- Krok 1: Obliczenie gradientu dla ostatniej warstwy ---
        
        if self.output_activation_name == 'softmax':
            dz = y_pred - y_true
        else:
            dz = (y_pred - y_true) # Działa też dla Liniowa + MSE
        
        # Pętla wstecz od ostatniej warstwy do pierwszej
        for i in reversed(range(len(self.weights))):
            
            # --- Krok 2: Obliczenie gradientów dW i db dla bieżącej warstwy ---
            
            # Aktywacja z *poprzedniej* warstwy (wejście do tej warstwy)
            a_prev = self.a[i] 
            
            # dW = (1/m) * (a_prev.T @ dz)
            dW = np.dot(a_prev.T, dz) / m
            
            # db = (1/m) * sum(dz wzdłuż osi batcha)
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            # --- Krok 3: Obliczenie gradientu dz dla *poprzedniej* warstwy ---
            
            # (Tylko jeśli to nie jest pierwsza warstwa)
            if i > 0:
                # z (przed aktywacją) dla poprzedniej warstwy
                z_prev = self.z[i - 1] 
                # Wagi bieżącej warstwy
                W_curr = self.weights[i]
                
                # Reguła łańcuchowa:
                # dz_prev = (dz_curr @ W_curr.T) * f'(z_prev)
                dz = np.dot(dz, W_curr.T) * self.activation_derivative(z_prev)

            # --- Krok 4: Aktualizacja wag i biasów (Gradient Descent) ---
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    # --- Metody Publiczne (API) ---

    def fit(self, X, y, epochs=1000, print_every=100):
        """Trenuje model na danych X, y przez zadaną liczbę epok."""
        loss_history = []
        for epoch in range(epochs):
            # 1. Propagacja do przodu
            y_pred = self._forward(X)
            
            # 2. Propagacja wsteczna (oblicza gradienty i aktualizuje wagi)
            self._backward(X, y)
            
            # 3. Oblicz i zapisz stratę (opcjonalnie)
            if epoch % print_every == 0:
                if self.output_activation_name == 'softmax':
                    loss = self._compute_cross_entropy_loss(y, y_pred)
                    print(f'Epoch {epoch}, Loss: {loss:.4f}')
                    loss_history.append(loss)
        return loss_history

    def predict_proba(self, X):
        """Zwraca "prawdopodobieństwa" (wyjście softmax) dla klas."""
        return self._forward(X)

    def predict(self, X):
        """Zwraca przewidziane etykiety klas (indeks o największym prawd.)."""
        probabilities = self.predict_proba(X)
        # np.argmax po osi 1 (dla każdego wiersza/próbki)
        return np.argmax(probabilities, axis=1)

    def evaluate(self, X, y_true):
        """Oblicza dokładność (accuracy) modelu na danych testowych."""
        predictions = self.predict(X)
        
        # Założenie: y_true jest zakodowane "one-hot" (np. [0, 0, 1])
        if y_true.ndim == 2:
            labels = np.argmax(y_true, axis=1)
        else:
            labels = y_true 
            
        accuracy = np.mean(predictions == labels)
        return accuracy

    def save_model(self, file_path):
        """Zapisuje wagi i biasy oraz konfigurację modelu do pliku .npz."""
        num_layers = len(self.weights)
        weights_to_save = np.empty(num_layers, dtype=object)
        biases_to_save = np.empty(num_layers, dtype=object)
        
        for i in range(num_layers):
            weights_to_save[i] = self.weights[i]
            biases_to_save[i] = self.biases[i]

        np.savez(file_path, 
                 weights=weights_to_save, 
                 biases=biases_to_save,
                 layer_sizes=self.layer_sizes,
                 activation=self.activation_name,
                 output_activation=self.output_activation_name,
                 allow_pickle=True) 

    @classmethod
    def load_model(cls, file_path):
        """
        Ładuje model z pliku .npz, działając jak konstruktor klasy.
        
        Użycie: model = MLP.load_model("sciezka/do/pliku.npz")
        """
        data = np.load(file_path, allow_pickle=True)
        
        # 1. Odczytaj konfigurację z pliku
        layer_sizes = data['layer_sizes']
        if layer_sizes.ndim == 0:
            layer_sizes = layer_sizes.item()
        # Konwersja na listę Pythona na wszelki wypadek
        layer_sizes = list(layer_sizes) 

        # Użyj .item(), aby wyodrębnić stringi ze skalarów numpy
        activation = data['activation'].item()
        output_activation = data['output_activation'].item()
        
        # 2. Stwórz nową instancję klasy (cls to odnośnik do MLP)
        # Wywoła to __init__ i ustawi domyślne (losowe) wagi
        model = cls(layer_sizes=layer_sizes,
                    activation=activation,
                    output_activation=output_activation)
        
        # 3. Nadpisz domyślne wagi tymi załadowanymi z pliku
        model.weights = list(data['weights'])
        model.biases = list(data['biases'])
        
        print("Model załadowany pomyślnie.")
        
        # 4. Zwróć gotowy obiekt modelu
        return model
