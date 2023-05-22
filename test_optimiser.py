import random
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from bayesian_optimizer import BayesianOptimiser, OptimiserArgument

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=False,
)
test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=False,
)


class Network(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, activation, dropout, *args, **kwargs):
        super().__init__()
        self.activation = activation
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_size1),
            self._activation_func(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size1, hidden_size2),
            self._activation_func(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size2, 10),

        )

    def forward(self, x):
        return self.layers(x)

    def _activation_func(self):
        if self.activation == "tanh":
            return nn.Tanh()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        else:
            # Default to ReLU
            return nn.ReLU()


def train_network(network, device, epochs, learning_rate, train_loader):
    network.train()

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            b_x = images.to(device)
            b_y = labels.to(device)
            output = network(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test_network(network, device, test_loader):
    network.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            test_output = network(images.to(device))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels.to(device)).sum().item()
            total += len(labels)
        return correct / total


def train(**kwargs):
    print("Testing args: " + str(kwargs))

    device = torch.device("cuda")
    network = Network(**kwargs)
    network.to(device)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs["batch_size"], shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=kwargs["batch_size"], shuffle=True, num_workers=1)

    train_network(network, device, kwargs["epochs"], kwargs["learning_rate"], train_loader)

    accuracy = test_network(network, device, test_loader)
    size = sum(np.prod(param.shape) for param in network.parameters())

    score = accuracy
    metrics = {"accuracy": accuracy, "size": size}

    print("Metrics: " + str(metrics))
    print()

    return score, metrics


if __name__ == '__main__':
    opt = BayesianOptimiser([
        OptimiserArgument("hidden_size1", [16, 32, 64, 128, 256]),
        OptimiserArgument("hidden_size2", [16, 32, 64, 128, 256]),
        OptimiserArgument("activation", ["relu", "sigmoid", "tanh"], categorical=True),
        OptimiserArgument("dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        OptimiserArgument("epochs", [1, 2, 3, 4]),
        OptimiserArgument("learning_rate", [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]),
        OptimiserArgument("batch_size", [16, 32, 64, 128])
    ], train)
    previous_time = time.time()
    opt.initialise(15)
    print(f"Time taken: {time.time() - previous_time}")
    opt.save("pre_initialised")

    ucb_lambda_options = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    length_scale_options = [1, 2, 3, 4, 5, 6, 7, 8]
    signal_variance_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    noise_scale_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in range(30):
        start_time = time.time()

        print(f"Iteration #{i}")

        opt = BayesianOptimiser.load("pre_initialised")

        ucb_lambda = random.choice(ucb_lambda_options)
        length_scale = random.choice(length_scale_options)
        signal_variance = random.choice(signal_variance_options)
        noise_scale = random.choice(noise_scale_options)

        print(f"UCB Lambda: {ucb_lambda}, "
              f"Length Scale: {length_scale}, "
              f"Signal Variance: {signal_variance}, "
              f"Noise Scale: {noise_scale}")
        print()

        opt.ucb_lambda = ucb_lambda
        opt.length_scale = length_scale
        opt.signal_variance = signal_variance
        opt.noise_scale = noise_scale

        print()
        print(f"Time taken: {round(time.time() - start_time)}s")
        print()
        print("=" * 200)
        print()





