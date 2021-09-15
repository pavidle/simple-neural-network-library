import numpy as np
from network import NeuralNetwork, Dense, Tanh, Model, Relu, Softmax, flatten
from tensorflow.keras.datasets import fashion_mnist, mnist
import matplotlib.pyplot as plt


def draw_fashion_with_predictions(test_choice, prediction):
    class_names = [
        'Футболка/Топ', 'Брюки', 'Свитер', 'Платье', 'Куртка',
        'Сандали/туфли', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки'
    ]
    plt.figure()
    plt.imshow(test_choice)
    plt.colorbar()
    plt.grid(False)
    plt.suptitle(f"Предположительно, это {class_names[np.argmax(prediction)]}", fontsize=20, fontweight='bold')
    plt.show()


def draw_digits_with_predictions(test_choice, prediction):
    plt.figure()
    plt.imshow(test_choice)
    plt.colorbar()
    plt.grid(False)
    plt.suptitle(f"Предположительно, это {np.argmax(prediction)}", fontsize=20, fontweight='bold')
    plt.show()


if __name__ == "__main__":
    nn = NeuralNetwork()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train_flatten, y_train_flatten = flatten(x_train, y_train, 28 * 28, 10)
    x_test_flatten, y_test_flatten = flatten(x_test, y_test, 28 * 28, 10)

    x_train_flatten /= 255
    x_test_flatten /= 255

    nn.add_layer(Dense(28 * 28, 40, activator=Tanh))
    nn.add_layer(Dense(40, 10, activator=Tanh))
    # nn.train_with_teacher(x_train_flatten[:10000], y_train_flatten[:10000], 25)
    model = Model(nn, "weights_digits.txt")
    # model.save()
    count_test = 100
    predictions = model.predict(x_test_flatten[:count_test])
    accuracy = model.get_accuracy(x_test_flatten[:count_test], y_test_flatten[:count_test])
    print(f"test accuracy: {accuracy}")

    for i in range(count_test):
        draw_digits_with_predictions(x_test[i], predictions[i])

    # for i in range(count_test):
    #     draw_fashion_with_predictions(x_test_without_shape[i], predictions[i])
