import numpy as np
from funkcje import compute_cross_entropy_loss

class MLP:
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate

    def add_layer(self, layer):
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
    
    def fit(self, X, y, epochs):
        loss_history = []
        for epoch in range(epochs):
            self.backward(X, y)
            if epoch % 500 == 0:
                y_pred = self.forward(X)
                loss = compute_cross_entropy_loss(y, y_pred)
                loss_history.append(loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
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

        
    
