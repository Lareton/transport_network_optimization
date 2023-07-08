import numpy as np


class TransportProblem:

    def __init__(self, T, lambda_, mu, l, w, t, sigma_star, gamma=0.01):
        """
        :param T: Tij
        :param lambda_: Lambda
        """
        self.T = T
        self.gamma = gamma
        self.lambda_ = lambda_
        self.mu = mu
        self.l = l
        self.w = w
        self.t = t
        # TODO: Set sigma star
        self.sigma_star = sigma_star

    def log_sum(self) -> float:
        return np.log(
            np.sum(np.exp((np.sum(-self.T + self.lambda_[None, ...], axis=1) + self.mu) / self.gamma)))

    def sigma_star_sum(self) -> np.array:
        return np.sum(self.sigma_star(self.t))

    def calc(self) -> float:
        return self.gamma * self.log_sum() - (self.l @ self.lambda_) - (self.w @ self.mu) + self.sigma_star_sum()

