import siec as sc
import warstwy as wr
import funkcje as fa
import numpy as np
import minst as mn
import mlp

def create_model(input_size,output_size=10):
    model = sc.MLP(learning_rate=0.1)
    model.add_layer(wr.Warstwa(input_size=input_size, layer_size=128, activation=fa.relu, activation_prime=fa.relu_derivative, weights_init='he'))
    model.add_layer(wr.Warstwa(input_size=128, layer_size=64, activation=fa.relu, activation_prime=fa.relu_derivative, weights_init='he'))
    model.add_layer(wr.Warstwa(input_size=64, layer_size=output_size, activation=fa.softmax, activation_prime=fa.derivative_one, weights_init='he'))
    return model
def rozdzielenie_danych(dane):
    dane_X = []
    labels = []
    for i in range(len(dane)):
        dane_X.append(dane[i][1])
        temp = np.zeros(10)
        temp[dane[i][0]] = 1
        labels.append(temp)
    return np.array(dane_X), np.array(labels)

dataset_treningowy = mn.create_dataset(rozmiar=300)
dataset_treningowy = mn.mix_dataset(dataset_treningowy)
dataset_testowy_1 = mn.mix_dataset(mn.create_dataset(rozmiar=50,start=200))
dataset_testowy_2 = mn.mix_dataset(mn.create_dataset(rozmiar=50,start=250))
dataset_testowy_3 = mn.mix_dataset(mn.create_dataset(rozmiar=50,start=300))
dataset_testowy_wszystkie = mn.mix_dataset(mn.create_dataset(rozmiar=1000))
dataset_testowy_pozostałe = mn.mix_dataset(mn.create_dataset(rozmiar=800,start=200))

model = create_model(784)
save_path = 'model_mnist.npz'
dane_wejsciowe, labels = rozdzielenie_danych(dataset_treningowy)

model_stary = mlp.MLP([784, 128, 64, 10], learning_rate=0.1)
model_stary.fit(dane_wejsciowe, labels, epochs=1000, print_every=500)
model.fit(dane_wejsciowe, labels, epochs=1000, print_every=500)
model.save_model(save_path)
for i, test_set in enumerate([dataset_treningowy,dataset_testowy_1, dataset_testowy_2, dataset_testowy_3,dataset_testowy_wszystkie,dataset_testowy_pozostałe], start=1):
    test_X, test_y = rozdzielenie_danych(test_set)
    accuracy = model.evaluate(test_X, test_y)
    accuracy_stary = model_stary.evaluate(test_X, test_y)
    print(f"Dokładność (stary model) na zestawie testowym {i}: {accuracy_stary * 100}%")
    print(f"Dokładność na zestawie testowym {i}: {accuracy * 100}%")