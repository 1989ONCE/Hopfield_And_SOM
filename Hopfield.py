import random, time
from sklearn.model_selection import train_test_split
import numpy as np

class Hopfield:
    def __init__(self):
        self.weights = None

    def setData(self, data):
        self.data = data

    def flatten_data(self, data):
        return np.array([np.array(x).flatten() for x in data])
    
    def add_noise(self, testing_data, noise_rate):
        noisy_data_list = []
        
        for data in testing_data:
            noisy_data = []
            for i in range(len(data)):
                noisy = data[i].copy()
                # get the index of the neurons to flip
                num_flip = int(len(noisy) * noise_rate)
                flip_index = random.sample(range(len(data[i])), num_flip)
                for i in flip_index:
                    noisy[i] = -1 if noisy[i] == 1 else 1
                noisy_data.append(noisy)
            noisy_data_list.append(noisy_data)
        return noisy_data_list

    def train(self):
        # Flatten input data
        flattened_data = self.flatten_data(self.data)
        num_patterns, num_neurons = flattened_data.shape

        # Initialize weight matrix
        self.weights = np.zeros((num_neurons, num_neurons))

        # Compute the weight matrix using the given formula
        for pattern in flattened_data:
            self.weights += np.outer(pattern, pattern)

        # Normalize weights and subtract identity matrix
        self.weights /= num_neurons
        self.weights -= (num_patterns / num_neurons) * np.eye(num_neurons)

        # Ensure no self-connections
        np.fill_diagonal(self.weights, 0)
        return self.weights

    def recall(self, test_pattern, max_iter, update_callback=None):
        # Copy the test pattern to avoid modifying the original
        shape = np.array(test_pattern).shape
        pattern = np.array(test_pattern).flatten()
        
        new_pattern = pattern.copy()
        # recall the pattern
        for _ in range(max_iter):
            # Compute the new pattern
            pattern = new_pattern
            new_pattern = np.dot(self.weights, pattern)
            new_pattern[new_pattern >= 0] = 1
            new_pattern[new_pattern < 0] = -1

            if update_callback:
                time.sleep(1)
                update_callback({_+1}, np.array(new_pattern.reshape(shape)))

            
            # Check if the pattern has converged
            if np.array_equal(new_pattern, pattern):
                print('Converged')
                break
        return new_pattern.reshape(shape)
            
            