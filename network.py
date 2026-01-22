# plik: network.py
import numpy as np
import pickle

class MLP:
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss_func = None
        self.loss_prime = None

    def add(self, layer):
        if len(self.layers) == 0:
            if layer.input_dim is None:
                raise ValueError("Pierwsza warstwa musi mieć input_dim!")
            layer.build(layer.input_dim)
        else:
            prev_output_dim = self.layers[-1].units
            layer.build(prev_output_dim)
        self.layers.append(layer)

    def compile(self, loss_func, loss_prime):
        self.loss_func = loss_func
        self.loss_prime = loss_prime

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epochs, batch_size=32, validation_data=None, patience=None, verbose=True):
        history = {'loss': [], 'val_loss': []}
        m = X.shape[0]
        
        best_loss = np.inf
        wait = 0
        
        for epoch in range(epochs):
            # Tasowanie
            perm = np.random.permutation(m)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            # Batch training
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                
                # 1. Forward
                y_pred = self.forward(X_batch)
                
                # 2. Backward
                error = self.loss_prime(y_batch, y_pred)
                for layer in reversed(self.layers):
                    error = layer.backward(error, self.learning_rate)

            # Raportowanie i Walidacja
            # Obliczamy stratę na zbiorze treningowym
            sample_pred = self.forward(X)
            loss = self.loss_func(y, sample_pred)
            history['loss'].append(loss)
            
            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.forward(X_val)
                val_loss = self.loss_func(y_val, val_pred)
                history['val_loss'].append(val_loss)
            
            if verbose:
                msg = f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)
            
            # Early Stopping
            if patience is not None:
                current_loss = val_loss if val_loss is not None else loss
                if current_loss < best_loss:
                    best_loss = current_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
        
        return history

    # --- POPRAWKA: Tego brakowało ---
    def predict(self, X):
        """Zwraca predykcje sieci (forward pass)."""
        return self.forward(X)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
