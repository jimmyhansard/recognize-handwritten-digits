import numpy as np

print('Recognize written digits with neural network!')

# Contains the number of neurons in the respective layers
sizes = [720, 1200, 10]

biases = [np.random.randn(y, 1) for y in sizes[1:]]
print(biases[0].shape)
print(biases[1].shape)
print(biases[0][1])


class NeuralNetwork(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]