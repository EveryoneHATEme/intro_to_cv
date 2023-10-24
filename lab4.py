import pickle
import tarfile
from random import choices, choice
from math import inf
from itertools import product, cycle
import numpy as np
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

# %%
np.random.seed(321)


# %% md
# Downloading the dataset
# %%
def unpickle(file):
    """
    source: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


# %% md
# "Training" the model or preprocessing the dataset
# %%
def process_image(data: np.ndarray) -> np.ndarray:
    """

    takes image as a row of 3072 elements,
    where the first 1024 entries contain the red channel values,
    the next 1024 the green, and the final 1024 the blue.

    returns image as an array of shape (32, 32, 3)

    """
    red_channel, green_channel, blue_channel = np.split(data, 3)
    pixels = np.stack((red_channel, green_channel, blue_channel), axis=-1)
    return np.reshape(pixels, (32, 32, 3))


# %%
def process_batch(batch: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    """

    processes a batch and returns a tuple of a list of processed images with shape (32, 32, 3) and list of labels

    """
    batch_images = list(map(process_image, batch[b'data']))
    batch_labels = batch[b'labels']

    return batch_images, batch_labels


# %%
batch_filenames = ['cifar-10-batches-py/data_batch_1',
                   'cifar-10-batches-py/data_batch_2',
                   'cifar-10-batches-py/data_batch_3',
                   'cifar-10-batches-py/data_batch_4',
                   'cifar-10-batches-py/data_batch_5']

train_images, raw_train_labels = [], []
for filename in batch_filenames:
    current_batch = unpickle(filename)
    images, labels = process_batch(current_batch)
    train_images.extend(images)
    raw_train_labels.extend(labels)


# %%
def one_hot_encoder(labels: list[int]) -> list[np.ndarray]:
    encoded: list[np.ndarray] = []
    for label in labels:
        encoded.append(np.zeros(10))
        encoded[-1][label] = 1.0
    return encoded


# %%
train_labels = one_hot_encoder(raw_train_labels)
# %%
test_batch = unpickle('cifar-10-batches-py/test_batch')
test_images, raw_test_labels = process_batch(test_batch)
# %%
test_labels = one_hot_encoder(raw_test_labels)
# %%
meta_data = unpickle('cifar-10-batches-py/batches.meta')
label_names = list(map(bytes.decode, meta_data[b'label_names']))
# %%
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray, train=False) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        pass


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)

        self.input: np.ndarray = np.array([])
        self.output: np.ndarray = np.array([])

    def __call__(self, x: np.ndarray, train=False) -> np.ndarray:
        if train:
            self.input = x
            self.output = x @ self.weights + self.bias
            return self.output
        else:
            return x @ self.weights + self.bias

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        input_error = error @ self.weights.T
        weights_error = self.input.T @ error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(error, axis=0)
        return input_error


# %%
class Activation(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray, train=False) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        pass


class Sigmoid(Activation):
    def __call__(self, x: np.ndarray, train=False) -> np.ndarray:
        if train:
            self.output = np.piecewise(
                x,
                [x < 0, x >= 0],
                [lambda a: np.exp(a) / (1 + np.exp(a)), lambda a: 1 / (1 + np.exp(-a))]
            )
            return self.output
        else:
            return np.piecewise(
                x,
                [x < 0, x >= 0],
                [lambda a: np.exp(a) / (1 + np.exp(a)), lambda a: 1 / (1 + np.exp(-a))]
            )

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        return error * self.output * (1 - self.output)


class Softmax(Activation):
    def __call__(self, x: np.ndarray, train=False, normalized=True) -> np.ndarray:
        # print(f'{x=}')
        if normalized:
            exponents = np.exp(x - np.max(x, axis=1, keepdims=True))
        else:
            exponents = np.exp(x)

        probabilities = exponents / np.sum(exponents, axis=1, keepdims=True)
        # print(f'{probabilities=}')

        return probabilities

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        return error  # only if it comes before cross entropy


# %%
class Loss(ABC):
    @abstractmethod
    def __call__(self, predicted: np.ndarray, actual: np.ndarray):
        pass

    @abstractmethod
    def get_gradient(self):
        pass


class CrossEntropyLoss(Loss):
    def __init__(self):
        self.gradient = None

    def __call__(self, predicted: np.ndarray, actual: np.ndarray):
        log_predicted = np.log(np.clip(predicted, 1e-7, 1 - 1e-7))
        self.gradient = (predicted - actual) / len(actual)
        return -np.sum(actual * log_predicted, axis=1)

    def get_gradient(self):
        return self.gradient


# %%
class Model:
    def __init__(self,
                 layers: list[Layer | Activation],
                 loss: Loss):
        self.layers = layers
        self.loss = loss

    def __call__(self, x: np.ndarray, train=False) -> np.ndarray:
        result = x
        for layer in self.layers:
            result = layer(result, train=train)
        return result

    def backpropagation(self, learning_rate=0.001):
        error = self.loss.get_gradient()
        for layer in self.layers[::-1]:
            error = layer.backward(error, learning_rate)

    def fit(self, x, y, epochs, learning_rate=0.001):
        for i in range(epochs):
            loss = 0
            for batch_x, batch_y in zip(x, y):
                output = self(batch_x, train=True)
                loss += np.mean(self.loss(output, batch_y))
                self.backpropagation(learning_rate)
            print(f'epoch: {i} loss: {loss / len(x)}')


# %%
train_images = np.array(list(map(np.ndarray.flatten, train_images))) / 255.0
train_labels = np.array(train_labels)
# %%
train_images_batched = np.array_split(train_images, train_images.shape[0] // 64 + 1)
train_labels_batched = np.array_split(train_labels, train_images.shape[0] // 64 + 1)
# %%
model = Model([
    Linear(3072, 1024),
    Sigmoid(),
    Linear(1024, 1024),
    Sigmoid(),
    Linear(1024, 10),
    Softmax()],
    loss=CrossEntropyLoss()
)
# %%
model.fit(train_images_batched, train_labels_batched, 5, learning_rate=0.1)
# %%
test_images = list(map(np.ndarray.flatten, test_images))
# %%
correct = 0
for image, label in zip(test_images, test_labels):
    prediction = model(np.array([image]) / 255.)
    print(prediction.argmax(), label.argmax())
    if prediction.argmax() == label.argmax():
        correct += 1
print(f'correct_predictions: {correct}/{len(test_images)}\naccuracy: {correct / len(test_images)}')