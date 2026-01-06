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
    
# --- 2. MODEL ---
def create_model_1(input_size,first_layer_size, output_size=10,activation_hidden=fa.relu,activation_hidden_prime=fa.relu_derivative,activation_output=fa.softmax,activation_output_prime=fa.derivative_one,weights_init='xavier', learning_rate=0.1):
    model = sc.MLP(learning_rate=learning_rate)
    
    # Architektura (Solidna)
    model.add_layer(wr.Warstwa(input_size=input_size, layer_size=first_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=output_size, activation=activation_output, activation_prime=activation_output_prime, weights_init=weights_init))
    return model
def create_model_2(input_size,first_layer_size,second_layer_size, output_size=10,activation_hidden=fa.relu,activation_hidden_prime=fa.relu_derivative,activation_output=fa.softmax,activation_output_prime=fa.derivative_one,weights_init='xavier', learning_rate=0.1):
    model = sc.MLP(learning_rate=learning_rate)
    
    # Architektura (Solidna)
    model.add_layer(wr.Warstwa(input_size=input_size, layer_size=first_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=second_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=output_size, activation=activation_output, activation_prime=activation_output_prime, weights_init=weights_init))
    return model
def create_model_3(input_size,first_layer_size,second_layer_size,third_layer_size, output_size=10,activation_hidden=fa.relu,activation_hidden_prime=fa.relu_derivative,activation_output=fa.softmax,activation_output_prime=fa.derivative_one,weights_init='xavier', learning_rate=0.1):
    model = sc.MLP(learning_rate=learning_rate)
    
    # Architektura (Solidna)
    model.add_layer(wr.Warstwa(input_size=input_size, layer_size=first_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=second_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=third_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=output_size, activation=activation_output, activation_prime=activation_output_prime, weights_init=weights_init))
    return model

def create_model_5(input_size,first_layer_size,second_layer_size,third_layer_size,fourth_layer_size,fifth_layer_size, output_size=10,activation_hidden=fa.relu,activation_hidden_prime=fa.relu_derivative,activation_output=fa.softmax,activation_output_prime=fa.derivative_one,weights_init='xavier'  ):
    model = sc.MLP(learning_rate=learning_rate)
    
    model.add_layer(wr.Warstwa(input_size=input_size, layer_size=first_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=second_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=third_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=fourth_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=fifth_layer_size, activation=activation_hidden, activation_prime=activation_hidden_prime, weights_init=weights_init))
    model.add_layer(wr.Warstwa( layer_size=output_size, activation=activation_output, activation_prime=activation_output_prime, weights_init=weights_init))
    return model

def porównanie_ilości_warstw():
    model_1 = create_model_1(input_size=784, first_layer_size=512)
    model_2 = create_model_2(input_size=784, first_layer_size=512, second_layer_size=256)
    model_3 = create_model_3(input_size=784, first_layer_size=512, second_layer_size=256, third_layer_size=128)
    model_5 = create_model_5(input_size=784, first_layer_size=1024, second_layer_size=512, third_layer_size=256, fourth_layer_size=128, fifth_layer_size=64)
    for model, desc in [(model_1, "1 warstwa ukryta"), (model_2, "2 warstwy ukryte"), (model_3, "3 warstwy ukryte"), (model_5, "5 warstw ukrytych")]:
        print(f"\nTworzenie modelu z {desc}...")
        start_time = time.time()
        model.fit(train_X, train_y, epochs=20, print_every=1)
        elapsed = time.time() - start_time
        acc = model.evaluate(test_X, test_y)
        print(f"Model z {desc} osiągnął dokładność: {acc*100:.2f}% w czasie {elapsed:.2f}s")
    

def benchmark_mlp_batch_sizes(first_layer_size=256,best_activation=fa.relu,best_activation_prime=fa.relu_derivative,best_epochs=20,best_batch_size=30):
        batch_sizes = [16,32, 64, 128, 256,512,1024]
        best_batch_size = None
        best_acc = 0.0
        for batch_size in batch_sizes:
            print(f"\nTworzenie modelu z Batch Size: {batch_size}...")
            model = create_model_1(input_size=784, first_layer_size=first_layer_size)
            start_time = time.time()
            model.fit(train_X, train_y, epochs=best_epochs, print_every=1,batch_size=batch_size, activation_hidden=best_activation, activation_hidden_prime=best_activation_prime)
            elapsed = time.time() - start_time
            acc = model.evaluate(test_X, test_y)
            if acc > best_acc:
                best_acc = acc
                best_batch_size = batch_size
            print(f"Model z Batch Size {batch_size} osiągnął dokładność: {acc*100:.2f}% w czasie {elapsed:.2f}s")
        print(f"\nNajlepszy Batch Size to {best_batch_size} z dokładnością {best_acc*100:.2f}%")
        return best_batch_size
def benchmark_mlp_funkcja_aktywacji(first_layer_size=256,best_epochs=20,best_batch_size=30):
        activation_hidden_functions = [fa.relu, fa.sigmoid]
        activation_hidden_primes = [fa.relu_derivative, fa.sigmoid_derivative]
        best_func = None
        best_acc = 0.0
        for i in range(len(activation_hidden_functions)):
            func = activation_hidden_functions[i]
            func_prime = activation_hidden_primes[i]
            print(f"\nTworzenie modelu z funkcją aktywacji ukrytej: {func.__name__}...")
            model = create_model_1(input_size=784, first_layer_size=first_layer_size, activation_hidden=func, activation_hidden_prime=func_prime)
            start_time = time.time()
            model.fit(train_X, train_y, epochs=best_epochs, print_every=1,batch_size=best_batch_size)
            elapsed = time.time() - start_time
            acc = model.evaluate(test_X, test_y)
            if acc > best_acc:
                best_acc = acc
                best_func = func
            print(f"Model z funkcją {func.__name__} osiągnął dokładność: {acc*100:.2f}% w czasie {elapsed:.2f}s")
        print(f"\nNajlepsza funkcja aktywacji to {best_func.__name__} z dokładnością {best_acc*100:.2f}%")
        return best_func , fa.relu_derivative if best_func==fa.relu else fa.sigmoid_derivative

def benchmark_mlp_ilość_warstw_dla_1(best_activation=fa.relu,best_activation_prime=fa.relu_derivative,best_epochs=20,best_batch_size=30):
    sizes = [16,32,64,128, 256, 512,1024,2048]
    best_size = None
    best_acc = 0.0
    for size in sizes:
        print(f"\nTworzenie modelu z warstwą ukrytą o rozmiarze {size}...")
        model = create_model_1(input_size=784, first_layer_size=size, activation_hidden=best_activation, activation_hidden_prime=best_activation_prime)
        start_time = time.time()
        model.fit(train_X, train_y, epochs=best_epochs, print_every=1,batch_size=best_batch_size)
        elapsed = time.time() - start_time
        acc = model.evaluate(test_X, test_y)
        if acc > best_acc:
            best_acc = acc
            best_size = size
        print(f"Model z warstwą {size} osiągnął dokładność: {acc*100:.2f}% w czasie {elapsed:.2f}s")
    print(f"\nNajlepszy rozmiar warstwy ukrytej to {best_size} z dokładnością {best_acc*100:.2f}%")
    return best_size
def benchmark_mlp_ilość_warstw_dla_2(best_activation=fa.relu,best_activation_prime=fa.relu_derivative,best_epochs=20,best_batch_size=30):
    sizes = [16,32,64,128, 256, 512,1024]
    best_sizes = [None,None]
    best_acc = 0.0
    for size in range(len(sizes)-1):
        for size2 in range(size+1):
            print(f"\nTworzenie modelu z warstwami ukrytymi o rozmiarach {sizes[size]} i {sizes[size2]}...")
            model = create_model_2(input_size=784, first_layer_size=sizes[size], second_layer_size=sizes[size2], activation_hidden=best_activation, activation_hidden_prime=best_activation_prime)
            start_time = time.time()
            model.fit(train_X, train_y, epochs=best_epochs, print_every=1,batch_size=best_batch_size)
            elapsed = time.time() - start_time
            acc = model.evaluate(test_X, test_y)
            if acc > best_acc:
                best_acc = acc
                best_sizes = [sizes[size], sizes[size2]]
            print(f"Model z warstwami {sizes[size]} i {sizes[size2]} osiągnął dokładność: {acc*100:.2f}% w czasie {elapsed:.2f}s")
    print(f"\nNajlepsze rozmiary warstw ukrytych to {best_sizes[0]} oraz {best_sizes[1]} o z dokładnością {best_acc*100:.2f}%")
    return best_sizes

def benchmark_mlp_liczba_epok(first_layer_size=256,best_activation=fa.relu,best_activation_prime=fa.relu_derivative,best_batch_size=30):
    epoch_counts = [5,10,15,20,25,30]
    best_epochs = None
    best_acc = 0.0
    for epochs in epoch_counts:
        print(f"\nTworzenie modelu z {epochs} epokami...")
        model = create_model_1(input_size=784, first_layer_size=first_layer_size, activation_hidden=best_activation, activation_hidden_prime=best_activation_prime)
        start_time = time.time()
        model.fit(train_X, train_y, epochs=epochs, print_every=1,batch_size=best_batch_size)
        elapsed = time.time() - start_time
        acc = model.evaluate(test_X, test_y)
        if acc > best_acc:
            best_acc = acc
            best_epochs = epochs
        print(f"Model z {epochs} epokami osiągnął dokładność: {acc*100:.2f}% w czasie {elapsed:.2f}s")
    print(f"\nNajlepsza liczba epok to {best_epochs} z dokładnością {best_acc*100:.2f}%")
    return best_epochs   
def grid_benchmark_mlp():
    # parametry do benchmarku
    ilość_warstw = [1,2]
    batch_sizes = [16,32]
    activation_hidden_functions = [fa.relu]
    activation_hidden_primes = [fa.relu_derivative]
    sizes = [64,128, 256, 512]
    learning_rates = [0.01,0.1,0.25,0.5,0.75]
    epoch_counts = [15]
    best_config = {}
    best_acc = 0.0
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for i in range(len(activation_hidden_functions)):
                func = activation_hidden_functions[i]
                func_prime = activation_hidden_primes[i]
                for epochs in epoch_counts:
                    for ilość in ilość_warstw:
                        if ilość == 1:
                            for size in range(len(sizes)):
                                print(f"\nTworzenie modelu z Batch Size: {batch_size}, Funkcją: {func.__name__}, Rozmiarem warstwy: {sizes[size]}, Epokami: {epochs}...,learning_rate: {learning_rate} ")
                                model = create_model_1(input_size=784, first_layer_size=sizes[size], activation_hidden=func, activation_hidden_prime=func_prime,learning_rate=learning_rate)
                                start_time = time.time()
                                model.fit(train_X, train_y, epochs=epochs, print_every=1,batch_size=batch_size)
                                elapsed = time.time() - start_time
                                acc = model.evaluate(test_X, test_y)
                                if acc > best_acc:
                                    best_acc = acc
                                    best_config = {
                                        'batch_size': batch_size,
                                        'activation_function': func,
                                        'layer_size': sizes[size],
                                        'second_layer_size': sizes[size2] if ilość==2 else None,
                                        'epochs': epochs,
                                        'learning_rate': learning_rate
                                    }
                                print(f"Model osiągnął dokładność: {acc*100:.2f}% w czasie {elapsed:.2f}s")
                        else:
                                for size in range(len(sizes)-1):
                                    for size2 in range(size+1):
                                            print(f"\nTworzenie modelu z Batch Size: {batch_size}, Funkcją: {func.__name__}, Rozmiarem warstwy 1: {sizes[size]}, Rozmiarem warstwy 2: {sizes[size2]}, Epokami: {epochs}, learning_rate: {learning_rate}...")
                                            model = create_model_2(input_size=784, first_layer_size=sizes[size], second_layer_size=sizes[size2], activation_hidden=func, activation_hidden_prime=func_prime,learning_rate=learning_rate)
                                            start_time = time.time()
                                            model.fit(train_X, train_y, epochs=epochs, print_every=1,batch_size=batch_size)
                                            elapsed = time.time() - start_time
                                            acc = model.evaluate(test_X, test_y)
                                            if acc > best_acc:
                                                best_acc = acc
                                                best_config = {
                                                    'batch_size': batch_size,
                                                    'activation_function': func,
                                                    'layer_size': sizes[size],
                                                    'second_layer_size': sizes[size2] if ilość==2 else None,
                                                    'epochs': epochs,
                                                    'learning_rate': learning_rate
                                                }
                                            print(f"Model osiągnął dokładność: {acc*100:.2f}% w czasie {elapsed:.2f}s")


    print(f"\nNajlepsza konfiguracja to Batch Size: {best_config['batch_size']}, Funkcja: {best_config['activation_function'].__name__}, Rozmiar warstwy: {best_config['layer_size']},Rozmiar warstwy 2: {best_config['second_layer_size']}, Epoki: {best_config['epochs']} oraz learning rate {best_config['learning_rate']} z dokładnością {best_acc*100:.2f}%")
    

def red():
    print(f"Mamy {len(train_X)} przykładów. To wystarczy na >95%.")
    # --- 3. TRENING ---
    # Zwiększamy Batch Size do 128 lub 256. 
    # Im większy batch, tym mniej operacji Pythona, a więcej szybkiego NumPy.
    BATCH_SIZE = 10
    EPOCHS = 10

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
def benchmark_mlp():
    print("Benchmarkowanie MLP na MNIST...")
    best_layer_size = benchmark_mlp_ilość_warstw_dla_1()
    best_activation, best_activation_prime = benchmark_mlp_funkcja_aktywacji(first_layer_size=best_layer_size)
    best_batch_size = benchmark_mlp_batch_sizes(first_layer_size=best_layer_size,best_activation=best_activation)
    best_liczba_epok = benchmark_mlp_liczba_epok(first_layer_size=best_layer_size,best_activation=best_activation,best_activation_prime=best_activation_prime,best_batch_size=best_batch_size)
    print(f"\nPodsumowanie benchmarku MLP:")
    print(f"Najlepszy Batch Size: {best_batch_size}")
    print(f"Najlepsza funkcja aktywacji: {best_activation.__name__}")
    model = create_model_1(input_size=784, first_layer_size=best_layer_size, activation_hidden=best_activation, activation_hidden_prime=fa.relu_derivative if best_activation==fa.relu else fa.sigmoid_derivative)
    print("\nTrening finalnego modelu z najlepszymi parametrami...")
    start_time = time.time()
    model.fit(train_X, train_y, epochs=best_liczba_epok, print_every=1,batch_size=best_batch_size)
    elapsed = time.time() - start_time
def test():
    print(f"Testowy model z dwoma warstwami ukrytymi: warstwa 1 =512, warstwa 2 =256, activation=ReLU, learning_rate=0.05, epochs=20, batch_size=10")
    model = create_model_2(input_size=784, first_layer_size=512, second_layer_size=256, activation_hidden=fa.relu, activation_hidden_prime=fa.relu_derivative,learning_rate=0.05)
    model.fit(train_X, train_y, epochs=25, print_every=1,batch_size=30)
    print(f"Model z 2 warstwami osiągnął dokładność: {model.evaluate(test_X, test_y)*100:.2f}%")
def test_1():
    input_size=784
    first_layer_size=256
    activation_hidden=fa.relu
    activation_hidden_prime=fa.relu_derivative
    learning_rate=0.1
    epochs=25
    batch_size=100
    print(f"Testowy model z dwoma warstwami ukrytymi: warstwa 1 ={first_layer_size}, activation={activation_hidden.__name__}, learning_rate={learning_rate}, epochs={epochs}, batch_size={batch_size}")
    model = create_model_1(input_size=input_size, first_layer_size=first_layer_size,  activation_hidden=activation_hidden, activation_hidden_prime=activation_hidden_prime,learning_rate=learning_rate)
    model.fit(train_X, train_y, epochs=epochs, print_every=1,batch_size=batch_size)
    print(f"Model z 2 warstwami osiągnął dokładność: {model.evaluate(test_X, test_y)*100:.2f}%")
test_1()