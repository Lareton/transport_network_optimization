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

        # self.x = torch.tensor(np.random.random((self.x_dim)), requires_grad=True).float()
        # self.y = torch.tensor(np.random.random((self.y_dim)), requires_grad=True).float()
        self.b_vector = torch.Tensor(np.random.random(self.m))
        self.c_matrix = torch.Tensor(np.random.random((self.m, self.x_dim)))

        b = torch.Tensor(np.random.normal(size=(self.x_dim + self.y_dim, self.x_dim + self.y_dim)))
        self.a_matrix = b @ torch.transpose(b, 0, 1)


    def calc(self,x, y):
        x = torch.tensor(x, requires_grad=True)
        y = torch.tensor(y, requires_grad=True)

        x.retain_grad()
        y.retain_grad()

        x_y = torch.concatenate([x, y], dim=0)

        summand_1 = self.a_matrix @ x_y @ x_y * self.CONST
        summand_2 = self.gamma * torch.logsumexp((self.c_matrix @ x - self.b_vector) / self.gamma, dim=0)
        result = summand_1 + summand_2

        result.backward()

        x_grad = copy.deepcopy(x.grad)
        y_grad = copy.deepcopy(y.grad)

        x.grad.zero_()
        y.grad.zero_()

        return result, x_grad, y_grad


def get_matrix(m, d, lams):
    """Returns m x d matrix with given min(m, d) singular values"""
    assert len(lams) == min(m, d)
    transpose = True
    if m > d:
        m, d = d, m
        transpose = False

    U = np.random.rand(d, d)
    Qd, _ = np.linalg.qr(U)
    K = Qd[:d, :m]
    K = K @ np.diag(np.sqrt(lams))

    U = np.random.rand(m, m)
    Qm, _ = np.linalg.qr(U)

    A = K @ Qm
    if transpose:
        A = A.T

    return A


class TestProblem2:
    def __init__(self):
        na, La = 1000, 100
        A = get_matrix(na, na, np.linspace(0, La, na))
        self.A = A.T @ A
        self.a = np.random.random(na)

        nb, Lb = 1000, 20
        B = get_matrix(nb, nb, np.linspace(0, Lb, nb))
        self.B = B.T @ B
        self.b = np.random.random(nb)

    def calc(self, x, y):
        res =  0.5 * np.transpose(x) @ self.A @ x
        res += np.transpose(self.a) @ x
        res += 0.5 * np.transpose(y) @ self.B @ y
        res += np.transpose(self.b) @ y

        grad_x = self.A @ x + self.a
        grad_y = self.B @ y + self.b

        return res, grad_x, grad_y
