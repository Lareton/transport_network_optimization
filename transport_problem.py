import numpy as np
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar()
I = TypeVar()


@dataclass
class Params:
    T: float | I
    lambda_: T
    mu: T
    gamma: float | I
    l: float | I
    w: float
    f_flow: float | I
    mu_pow: float
    t_flow: float | I
    raw: float | I
    mu_pow: float | I


class Grad:

    def __init__(self, params):
        self.params = params

    def grad_T(self):
        ...

    def invert_tau(self, t):
        return (
                self.params.f_flow * t ** self.params.mu_pow - self.params.f_flow - self.params.t_flow ** self.params.mu_pow) / (
                self.params.t_flow ** self.params.mu_pow * self.params.raw)

    def d(self):
        return np.exp((-self.params.T + self.params.lambda_ + self.params.mu) / self.params.gamma) / np.sum(
            np.exp((-self.params.T + self.params.lambda_ + self.params.mu) / self.params.gamma))

    def grad_dF_dla(self):
        return np.sum(np.exp(-self.params.T + self.params.lambda_[None, ...] + self.params.mu[..., None]),
                      axis=1) / np.sum(
            np.exp(-self.params.T + self.params.lambda_[None, ...] + self.params.mu[..., None])) + self.params.l

    def grad_dF_dt(self):
        return self.params.d() * self.params.grad_T() + self.invert_tau(self.params.T)

    def grad_dF_dmu(self):
        return np.sum(np.exp(-self.params.T + self.params.lambda_[None, ...] + self.params.mu[..., None]),
                      axis=0) / np.sum(
            np.exp(-self.params.T + self.params.lambda_[None, ...] + self.params.mu[..., None])) + self.params.w


class TransportProblem:
    def __init__(self, params):
        self.params = params

    def sigma_star(self, t):
        return self.params.f_flow * (
                (t - self.params.t_flow) / (self.params.t_flow * self.params.raw)) ** self.params.mu_pow * (
                t - self.params.t_flow) / (1 + self.params.mu_pow)

    def log_sum(self) -> float:
        return np.log(
            np.sum(np.exp((np.sum(-self.params.T + self.params.lambda_[None, ...],
                                  axis=1) + self.params.mu) / self.params.gamma)))

    def sigma_star_sum(self) -> np.array:
        return np.sum(self.sigma_star(self.params.t))

    def calc(self) -> float:
        return self.params.gamma * self.log_sum() - (self.params.l @ self.params.lambda_) - (
                self.params.w @ self.params.mu) + self.sigma_star_sum()


def main():
    params = Params()


if __name__ == '__main__':
    main()
