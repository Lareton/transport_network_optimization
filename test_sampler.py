import torch
import numpy as np

class TestProblem:
    CONST = 0.5

    def __init__(self, gamma=0.01, m=100, x_dim=1000, y_dim=2000):
        self.m = m
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.gamma = gamma

        self.x = torch.tensor(np.random.random((self.x_dim)), requires_grad=True).float()
        self.y = torch.tensor(np.random.random((self.y_dim)), requires_grad=True).float()


    def calc(self):
        x_y = torch.concatenate([self.x, self.y], dim=0)
        b = torch.Tensor(np.random.normal(size=(self.x_dim + self.y_dim, self.x_dim + self.y_dim)))
        a_matrix = b * torch.transpose(b, 0, 1)
        b_vector = torch.Tensor(np.random.random(self.m))
        c_matrix = torch.Tensor(np.random.random((self.m, self.x_dim)))

        summand_1 = a_matrix @ x_y @ x_y * self.CONST
        summand_2 = self.gamma * torch.logsumexp((c_matrix @ self.x - b_vector) / self.gamma, dim=0)
        result = summand_1 + summand_2

        self.x.retain_grad()
        self.y.retain_grad()
        result.backward()

        x_grad = self.x.grad
        y_grad = self.y.grad

        return result, x_grad, y_grad