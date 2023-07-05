import torch
import numpy as np
import copy


class TestProblem:
    CONST = 0.5

    def __init__(self, gamma=0.01, m=100, x_dim=1000, y_dim=2000):
        self.m = m
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.gamma = gamma

        self.x = torch.tensor(np.random.random((self.x_dim)), requires_grad=True).float()
        self.y = torch.tensor(np.random.random((self.y_dim)), requires_grad=True).float()
        self.b = torch.Tensor(np.random.normal(size=(self.x_dim + self.y_dim, self.x_dim + self.y_dim)))
        self.b_vector = torch.Tensor(np.random.random(self.m))
        self.c_matrix = torch.Tensor(np.random.random((self.m, self.x_dim)))
        b = torch.Tensor(np.random.normal(size=(self.x_dim + self.y_dim, self.x_dim + self.y_dim)))
        self.a_matrix = b @ torch.transpose(b, 0, 1)

        self.x.retain_grad()
        self.y.retain_grad()

    def calc(self):
        x_y = torch.concatenate([self.x, self.y], dim=0)

        summand_1 = self.a_matrix @ x_y @ x_y * self.CONST
        summand_2 = self.gamma * torch.logsumexp((self.c_matrix @ self.x - self.b_vector) / self.gamma, dim=0)
        result = summand_1 + summand_2

        result.backward()

        x_grad = copy.deepcopy(self.x.grad)
        y_grad = copy.deepcopy(self.y.grad)

        self.x.grad.zero_()
        self.y.grad.zero_()

        return result, x_grad, y_grad
