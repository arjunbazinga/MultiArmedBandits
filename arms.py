import numpy as np


class Arm:

    def __init__(self, mu=1, sigma=1):
        self.mu = np.absolute(np.random.uniform())
        self.sigma = np.absolute(np.random.uniform())

    def pull(self):
        reward = np.random.normal(self.mu, self.sigma, 1)
        return reward


def get_arms(k):
    arms = []
    for i in range(k):
        arms.append(Arm())
    return arms
