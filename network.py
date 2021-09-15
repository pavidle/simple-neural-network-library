from abc import ABC, abstractmethod
import numpy as np
import typing as t
import ast
from keras.utils import np_utils


def _calculate_layer_values(
        input_data: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray
) -> np.ndarray:
    return np.dot(weights, input_data) + bias


def flatten(x, y, input_size, output_size):
    x_train = x.reshape(x.shape[0], input_size, 1).astype("float32")
    y_train = np_utils.to_categorical(y).reshape(y.shape[0], output_size, 1)
    return x_train, y_train


class LossFunction(ABC):
    @abstractmethod
    def calculate(self, true_values: np.ndarray, predict_values: np.ndarray):
        pass

    @abstractmethod
    def calculate_derivative(self, true_values: np.ndarray, predict_values: np.ndarray):
        pass


class MeanSquare(LossFunction):
    def calculate(self, true_values: np.ndarray, predict_values: np.ndarray) -> t.Any:
        return np.mean(pow(true_values - predict_values, 2))

    def calculate_derivative(self, true_values: np.ndarray, predict_values: np.ndarray) -> t.Any:
        return 2 * (predict_values - true_values) / np.size(true_values)


class Layer(ABC):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            activator: t.Type["Activator"]
    ):
        self._input = None
        self._weights = np.random.randn(output_size, input_size)
        self._bias = np.random.randn(output_size, 1)
        self._activator = activator()

    def get_activator(self) -> "Activator":
        return self._activator

    def get_weights(self) -> np.ndarray:
        return self._weights

    def get_biases(self) -> np.ndarray:
        return self._bias

    @abstractmethod
    def forward_on_signal(self, input_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward_on_signal(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        pass


class Activator(ABC):
    def __init__(self):
        self.__function = self._calculate
        self.__derivative = self._calculate_derivative
        self.__current_input = None
        super().__init__()

    def activate(self, input_data) -> list:
        self.__current_input = input_data
        return self.__function(self.__current_input)

    def activate_derivative(self, gradient) -> list:
        return np.multiply(gradient, self.__derivative(self.__current_input))

    @abstractmethod
    def _calculate(self, x):
        pass

    @abstractmethod
    def _calculate_derivative(self, x):
        pass


class Tanh(Activator):
    def _calculate(self, x) -> list:
        return np.tanh(x)

    def _calculate_derivative(self, x) -> list:
        return 1 - np.tanh(x) ** 2


class Relu(Activator):
    def _calculate(self, x):
        return x * (x > 0)

    def _calculate_derivative(self, x):
        return 1. * (x > 0)


class Softmax(Activator):
    def _calculate(self, x):
        tmp = x - x.max(axis=1).reshape(-1, 1)
        exp_tmp = np.exp(tmp)
        return exp_tmp / exp_tmp.sum(axis=1).reshape(-1, 1)

    def _calculate_derivative(self, x):
        sm = self._calculate(x)
        return -np.outer(sm, sm) + np.diag(sm.flatten())


class Dense(Layer):
    def forward_on_signal(self, input_data: np.ndarray) -> np.ndarray:
        self._input = np.array(input_data)
        return _calculate_layer_values(self._input, self._weights, self._bias)

    def backward_on_signal(self, gradient: np.ndarray, learning_rate: int) -> np.ndarray:
        gradient_weights = np.dot(gradient, self._input.T)
        self._weights -= learning_rate * gradient_weights
        self._bias -= learning_rate * gradient
        return np.dot(self._weights.T, gradient)


class NeuralNetwork:
    def __init__(self, layers: t.Optional[t.List[Layer]] = None):
        self.__layers: t.List[Layer] = layers or list()
        self.__error = MeanSquare()

    def add_layer(self, layer: Layer) -> None:
        self.__layers.append(layer)

    def get_layers(self) -> list:
        return self.__layers

    def train_with_teacher(
            self,
            input_data: list,
            output_data: list,
            epochs: int,
            learning_rate: float = 0.1,
    ) -> None:
        for epoch in range(epochs):
            error = 0
            for x, y in zip(input_data, output_data):
                o = x
                for layer in self.__layers:
                    o = layer.forward_on_signal(o)
                    o = layer.get_activator().activate(o)
                error += self.__error.calculate(y, o)
                gradient = self.__error.calculate_derivative(y, o)
                for layer in reversed(self.__layers):
                    gradient = layer.get_activator().activate_derivative(gradient)
                    gradient = layer.backward_on_signal(gradient, learning_rate)
            error /= len(input_data)
            print(f"epoch: {epoch + 1} / {epochs}, error: {error}")

    def predict(self, input_data) -> list:
        output = list()
        for x in input_data:
            o = x
            for layer in self.__layers:
                o = layer.forward_on_signal(o)
                o = layer.get_activator().activate(o)
            output.append(o.tolist())
        return output


class Model:
    def __init__(self, network: NeuralNetwork, path: str):
        self.__network = network
        self.__path = path

    def save(self) -> None:
        with open(self.__path, "w") as file:
            for layer in self.__network.get_layers():
                file.write(str(layer.get_weights().tolist()) + '\n\n')
                file.write(str(layer.get_biases().tolist()) + '\n\n')

    def get(self) -> list:
        with open(self.__path, "r") as file:
            return [
                np.array(
                    ast.literal_eval(data.replace("\n", "")))
                for data in file.read().split('\n\n') if data != ""
            ]

    def predict(self, input_data) -> list:
        model_data = self.get()
        layers = self.__network.get_layers()
        output = list()
        input_data = np.array(input_data)
        for x in input_data:
            o = x
            for index in range(0, len(model_data), 2):
                o = _calculate_layer_values(o, model_data[index], model_data[index + 1])
                o = layers[index - 1].get_activator().activate(o)
            output.append(o.tolist())
        return output

    def get_accuracy(self, x_test, y_test) -> float:
        predictions = self.predict(x_test)
        count_predict_correctly = 0
        count_test = len(x_test)
        for index, predict in enumerate(predictions):
            pred = np.argmax(predict)
            true_value = np.argmax(y_test[index])
            if pred == true_value:
                count_predict_correctly += 1
        return count_predict_correctly / count_test
