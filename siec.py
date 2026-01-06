import numpy as np
from funkcje import compute_cross_entropy_loss

class MLP:
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate

    def add_layer(self, layer):
        if len(self.layers) > 0:
            prev_layer = self.layers[-1]
            if layer.weights is None:
                layer.set(prev_layer.layer_size)
        self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    def backward(self, X, y_true):  
        y_pred = self.forward(X)
        m = y_true.shape[0]
        dz = (y_pred - y_true)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i > 0:
                prev_layer = self.layers[i - 1]
                z_prev = prev_layer.z
                activation_derivative = prev_layer.activation_prime(z_prev)
                dz_prev = np.dot(dz, layer.weights.T) * activation_derivative
            layer.backward(dz, self.learning_rate, m)
            if i > 0:
                dz = dz_prev
        
    def fit(self, X, y, epochs=1000, print_every=500,batch_size=None):
        loss_history = []
        n_samples = X.shape[0]

        if batch_size is None:
            batch_size = n_samples
        
        
        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                self.backward(X_batch, y_batch)
            
            # --- POPRAWKA ---
            # Sprawdzamy > 0 NA SAMYM POCZĄTKU.
            # Jeśli print_every to 0, cała reszta linii jest ignorowana.
            if print_every > 0 and epoch % print_every == 0:
                y_pred = self.forward(X)
                loss = compute_cross_entropy_loss(y, y_pred)
                loss_history.append(loss)
                print(f"Epoch {epoch}/{epochs}, Loss: {loss}")
                if loss < 0.001:
                    print("Wczesne zatrzymanie: strata spadła poniżej 0.01")
                    return loss_history
            # ----------------
            
        return loss_history    
    def predict_proba(self, X):
        y_pred = self.forward(X)
        return y_pred
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    def evaluate(self, X, y_true):
        predictions = self.predict(X)
        if y_true.ndim == 2:
            labels = np.argmax(y_true, axis=1)
        else:
            labels = y_true 
        accuracy = np.mean(predictions == labels)
        return accuracy
    def save_model(self, file_path):
        np.savez(file_path, layers=self.layers)
    @classmethod
    def load_model(cls, file_path):
        data = np.load(file_path, allow_pickle=True)
        cls.layers = data['layers'].tolist()
        
        model = cls()
        model.layers = cls.layers
        
        print(f"Model załadowany z {file_path}")
        return model

        
    
