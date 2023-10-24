import numpy as np
import pickle
import requests
import tarfile


# %%
def unpickle(file):
    """
    source: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


# %%
dataset = requests.get('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')

with open('cifar-10-python.tar.gz', 'wb') as dataset_file:
    dataset_file.write(dataset.content)
# %%
with tarfile.open('cifar-10-python.tar.gz') as tar:
    tar.extractall()


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
train_images = [image.reshape((-1, 1)) / 255. for image in train_images]
train_labels = [image.reshape((-1, 1)) for image in train_labels]

test_images = [image.reshape((-1, 1)) / 255. for image in test_images]
test_labels = [image.reshape((-1, 1)) for image in test_labels]


# %%
class Softmax:
    def __init__(self):
        self.output: np.ndarray = np.array([])

    def __call__(self, x: np.ndarray, normalization=True) -> np.ndarray:
        if normalization:
            exponents = np.exp(x - np.max(x))
        else:
            exponents = np.exp(x)

        self.output = exponents / np.sum(exponents)

        return self.output

    def backward(self, error: np.ndarray, learning_rate=0.001) -> np.ndarray:
        gradient = self.output * (np.eye(self.output.shape[0]) - self.output.T)
        return np.sum(error * gradient, axis=0).reshape(-1, 1)


# %%
class ReLU:
    def __init__(self):
        self.output: np.ndarray = np.array([])

    def __call__(self, x: np.ndarray):
        self.output = np.maximum(x, 0)
        return self.output

    def backward(self, error: np.ndarray, learning_rate=0.001):
        gradient = np.piecewise(self.output, [self.output <= 0, self.output > 0], [0, 1])
        return error * gradient


# %%
class Dense:
    def __init__(self, input_shape: int, output_shape: int):
        self.weights: np.ndarray = np.random.rand(output_shape, input_shape) * 2 - 1
        self.input: np.ndarray = np.array([])
        self.output: np.ndarray = np.array([])

    def __call__(self, x: np.ndarray):
        self.input = x
        self.output = self.weights.dot(x)
        return self.output

    def backward(self, error: np.ndarray, learning_rate=0.001):
        input_error = np.sum(error * self.weights, axis=0).reshape((-1, 1))
        self.weights -= learning_rate * error @ self.input.T
        return input_error


# %%
class CrossEntropyLoss:
    def __init__(self):
        self.gradient: np.ndarray = np.array([])

    def __call__(self, y_actual: np.ndarray, y_predicted: np.ndarray):
        self.gradient = -y_actual / (y_predicted + 1e-100)
        return -np.sum(y_actual * np.log(y_predicted + 1e-100))

    def get_gradient(self):
        return self.gradient


# %%
class Model:
    def __init__(self, layers: list, loss: CrossEntropyLoss):
        self.layers = layers
        self.loss = loss

    def __call__(self, x: np.ndarray):
        prediction = x.copy()
        for layer in self.layers:
            prediction = layer(prediction)
        return prediction

    def backpropagation(self, learning_rate: float = 0.001):
        error = self.loss.get_gradient()
        for layer in self.layers[::-1]:
            error = layer.backward(error, learning_rate)

    def fit(self, x: list[np.ndarray], y: list[np.ndarray], epochs: int = 1, learning_rate: float = 0.001):
        for i in range(epochs):
            loss = 0
            for _x, _y in zip(x, y):
                prediction = self(_x)
                loss += self.loss(_y, prediction)
                self.backpropagation(learning_rate)
            loss /= len(x)
            print(f'epoch: {i + 1}, average loss: {loss}')

    def test(self, x: list[np.ndarray], y: list[np.ndarray]):
        loss = 0
        correct_predictions = 0
        for _x, _y in zip(x, y):
            prediction = self(_x)
            loss += self.loss(_y, prediction)
            correct_predictions += 1 if prediction.argmax() == _y.argmax() else 0
        print(
            f'average loss: {loss / len(x)}, correct predictions: {correct_predictions}/{len(x)} ({correct_predictions / len(x) * 100:.2f}%)')


# %%
class Dense:
    def __init__(self, input_shape: int, output_shape: int):
        self.weights: np.ndarray = np.random.rand(output_shape, input_shape) * 2 - 1
        self.bias: np.ndarray = np.random.rand() * 2 - 0.5
        self.input: np.ndarray = np.array([])
        self.output: np.ndarray = np.array([])

    def __call__(self, x: np.ndarray):
        self.input = x
        self.output = self.weights.dot(x) + self.bias
        return self.output

    def backward(self, error: np.ndarray, learning_rate=0.001):
        input_error = np.sum(error * self.weights, axis=0).reshape((-1, 1))
        self.weights -= learning_rate * error @ self.input.T
        self.bias -= learning_rate * error
        return input_error


# %%
second_task_model = Model(
    [
        Dense(3072, 256),
        ReLU(),
        Dense(256, 10),
        Softmax()
    ],
    CrossEntropyLoss()
)
# %%
second_task_model.fit(train_images, train_labels, 100)
# %%
second_task_model.test(test_images, test_labels)