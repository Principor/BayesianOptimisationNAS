import numpy as np
import pandas as pd
import pickle
import plotly.express as px


def distance(x1, x2):
    # Distances are calculated using "taxicab distance" in so that they match the number of edits separating two sets
    # of arguments
    return np.sum(np.abs(x1-x2), axis=-1)


class OptimiserArgument:
    def __init__(self, name, values, categorical=False):
        self.name = name
        self.values = values
        self.categorical = categorical


class BayesianOptimiser:
    def __init__(self, arguments, sample_function):
        arguments = list(arguments)
        self._n_combinations = np.prod([len(argument.values) for argument in arguments])
        self._positions = np.array(self._generate_positions(arguments))
        self._argument_values = self._generate_argument_values(arguments)

        self._sample_function = sample_function
        self._n_samples = 0

        self._sampled_indices = []
        self._sampled_scores = []
        self._sampled_metrics = []

        self.ucb_lambda = 1.1
        self.length_scale = 5
        self.signal_variance = 1.0
        self.noise_scale = 0.5

    def initialise(self, n_points):
        print("Initialising...")
        for i in range(n_points):
            print(f"#{i}")
            self._sample_point(np.random.randint(0, self._n_combinations))

    def optimise(self, iterations):
        if self._n_samples < 2:
            raise Exception("At least 2 samples are needed to calculate the acquisition function")
        print("Optimising...")
        for i in range(iterations):
            print(f"#{i}")
            self._sample_point(self._get_highest_acquisition())

    def get_best_arguments(self):
        index = np.argmax(self._sampled_scores)
        args = self._argument_values[self._sampled_indices[index]]
        score = self._sampled_scores[index]
        return args, score

    def plot_scatter(self, x_axis, y_axis, log_x=False, log_y=False):
        sampled_args = [self._argument_values[i] for i in self._sampled_indices]
        frames = [pd.DataFrame.from_records(self._sampled_metrics), pd.DataFrame.from_records(sampled_args)]
        data = pd.concat(frames, axis=1, join='inner')
        fig = px.scatter(
            data, x=x_axis, y=y_axis, log_x=log_x, log_y=log_y, hover_data=data.columns.values
        )
        fig.show()

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    def _kernel(self, x1, x2):
        x1 = np.expand_dims(x1, 0)
        x2 = np.expand_dims(x2, 1)
        sqr_dist = np.square(distance(x1, x2))
        exp = np.exp(-sqr_dist / (2 * np.square(self.length_scale)))
        return np.square(self.signal_variance) * exp

    def _gaussian_process(self):
        adjusted_results = (self._sampled_scores - np.mean(self._sampled_scores)) / np.std(self._sampled_scores)

        max_indices = 5000
        if self._n_combinations > max_indices:
            test_indices = np.random.choice(self._n_combinations, max_indices)
            test_size = max_indices
        else:
            test_indices = np.arange(self._n_combinations)
            test_size = self._n_combinations

        sampled_positions = self._positions[self._sampled_indices]
        test_positions = self._positions[test_indices]

        noise = np.identity(sampled_positions.shape[0]) * self.noise_scale
        k = self._kernel(sampled_positions, sampled_positions)
        k_inv = np.linalg.inv(k + noise)
        k_star = self._kernel(sampled_positions, test_positions)
        k_star_t = np.transpose(k_star)
        k_star2 = self._kernel(test_positions, test_positions)

        mean = k_star @ k_inv @ adjusted_results
        covariance = k_star2 - k_star @ k_inv @ k_star_t

        return mean[-test_size:], covariance[-test_size:], test_indices

    def _get_highest_acquisition(self):
        mean, covariance, indices = self._gaussian_process()
        # Acquisition function is calculated using upper confidence bound
        std_error = np.diagonal(covariance)
        max_index = np.argmax(mean + std_error * self.ucb_lambda)
        return indices[max_index]

    def _sample_point(self, index):
        args = self._argument_values[index]
        score, metrics = self._sample_function(**args)

        self._sampled_indices.append(index)
        self._sampled_scores.append(score)
        self._sampled_metrics.append(metrics)

        self._n_samples += 1

    def _generate_argument_values(self, arguments, values_start=None):
        if values_start is None:
            values_start = {}
        if len(arguments) == 0:
            return [values_start]
        argument = arguments[0]
        argument_values = []
        for value in argument.values:
            new_values = {**values_start, argument.name: value}
            argument_values += self._generate_argument_values(arguments[1:], new_values)
        return argument_values

    def _generate_positions(self, arguments, position_start=None):
        if position_start is None:
            position_start = []
        if len(arguments) == 0:
            return [position_start]
        argument = arguments[0]
        positions = []
        for i in range(len(argument.values)):
            if argument.categorical:
                # Position of 0.5 is used to ensure that changing a categorical argument still has a distance
                # of 1 using "taxicab distance"
                position_ending = [0.5 if value == argument.values[i] else 0 for value in argument.values]
            else:
                position_ending = [i]
            new_position = position_start + position_ending
            positions += self._generate_positions(arguments[1:], new_position)
        return positions
