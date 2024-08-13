import numpy as np

learning_rate = 0.1
epochs = 10_000

np.random.seed(89798)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_of_sigmoid(x):
    return x * (1 - x)


class Neuron:
    def __init__(self):
        self._weights = np.random.uniform(size=2)
        self._bias = np.random.uniform()
        self._input = None

    def forward(self, x):
        self._input = x
        return sigmoid(x @ self._weights + self._bias)

    def backward(self, x, loss_):
        d = loss_ * derivative_of_sigmoid(x)
        self._weights -= learning_rate * self._input.T * d
        self._bias -= learning_rate * d.sum()
        return d

    @property
    def weights(self):
        return self._weights

    @property
    def input(self):
        return self._input


class Model:
    def __init__(self):
        self._hidden_neuron1 = Neuron()
        self._hidden_neuron2 = Neuron()
        self._output_neuron = Neuron()
        self._out = None

    def forward(self, x):
        o1 = self._hidden_neuron1.forward(x)
        o2 = self._hidden_neuron2.forward(x)
        x = np.array([o1, o2]).T
        o3 = self._output_neuron.forward(x)
        self._out = o3
        return self._out

    def backward(self, derivative_of_loss):
        d = self._output_neuron.backward(self._out, derivative_of_loss)
        self._hidden_neuron1.backward(self._output_neuron.input[0], d * self._output_neuron.weights[0])
        self._hidden_neuron2.backward(self._output_neuron.input[1], d * self._output_neuron.weights[1])


def main():
    train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    label = np.array([0, 1, 1, 0])
    model = Model()
    for epoch in range(epochs):
        for (example, answer) in zip(train, label):
            predicted = model.forward(example)
            err = (predicted - answer)**2
            model.backward(2 * (predicted - answer))
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}: loss = {err}')
    for example in train:
        predicted = model.forward(example)
        print(f'For example = {example}, predicted = {predicted}')


if __name__ == '__main__':
    main()
