import torch
import numpy as np
import copy
import cvxpy as cp


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


class TestProblem:
    CONST = 0.5

    def __init__(self, gamma=0.01, La=200, m=100, x_dim=100, y_dim=100,
                 svxpy_solver="SCS", svxpy_verbose=True, A_cond = 1e-3):
        self.m = m
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.La = La
        self.gamma = gamma

        # генерируем матрицы для задачи
        self.a_matrix_np = get_matrix(self.x_dim + self.y_dim, self.x_dim + self.y_dim,
                                    np.linspace(self.La * A_cond, self.La, self.x_dim + self.y_dim)) # b @ torch.transpose(b, 0, 1)
        self.a_matrix_np = self.a_matrix_np @ self.a_matrix_np.T
        self.a_matrix = torch.FloatTensor(self.a_matrix_np)

        self.b_vector_np = np.random.random(self.m)
        self.b_vector = torch.FloatTensor(self.b_vector_np)

        self.c_matrix_np = get_matrix(self.m, self.x_dim, np.linspace(1, 1, min(self.m, self.x_dim)))
        self.c_matrix = torch.FloatTensor(self.c_matrix_np)

        self.Lx = self.La + 1/gamma
        self.Ly = self.La

        # находим решение получившейся задачи
        f_star, x1_star, x2_star = self.find_solution_by_cvxpy(solver=svxpy_solver, verbose=svxpy_verbose)
        self.x1_star = x1_star
        self.x2_star = x2_star
        self.f_star = f_star

    def f1(self, x1, x2):
        x_y = torch.cat([torch.tensor(xi).float() for xi in [x1, x2]], dim=0)
        return (x_y.t() @ self.a_matrix @ x_y * self.CONST).detach().numpy()

    def f2(self, x1):
        return self.gamma * torch.logsumexp((self.c_matrix @ x1 - self.b_vector) / self.gamma, dim=0)

    def calc_by_one_block(self, vars):
        x = vars[:self.x_dim]
        y = vars[self.x_dim:]
        assert len(vars) == self.x_dim + self.y_dim

        res, grad_x, grad_y = self.calc(x, y)
        common_grad = np.hstack([grad_x, grad_y])
        return res, common_grad

    def calc(self, x, y):
        """Считаем значение тестовой функции и градиент"""
        x = torch.tensor(x, requires_grad=True).float()
        y = torch.tensor(y, requires_grad=True).float()

        x.retain_grad()
        y.retain_grad()

        x_y = torch.cat([x, y], dim=0)

        summand_1 = x_y.t() @ self.a_matrix @ x_y * self.CONST
        summand_2 = self.gamma * torch.logsumexp((self.c_matrix @ x - self.b_vector) / self.gamma, dim=0)
        result = summand_1 + summand_2

        result.backward()

        x_grad = copy.deepcopy(x.grad)
        y_grad = copy.deepcopy(y.grad)

        x.grad.zero_()
        y.grad.zero_()

        return result.detach().numpy(), x_grad.numpy(), y_grad.numpy()

    def find_solution_by_cvxpy(self, **kwargs):
        """Ищем решение готовым солвером (не факт что сходится)"""
        x = cp.Variable(self.x_dim)
        y = cp.Variable(self.y_dim)

        x_y = cp.hstack([x, y])

        problem = cp.Problem(
            cp.Minimize(self.CONST * cp.atoms.quad_form(x_y, np.array(self.a_matrix)) +
                        self.gamma * cp.atoms.log_sum_exp((self.c_matrix_np @ x - self.b_vector_np) / self.gamma)))

        min_value = problem.solve(**kwargs)

        return min_value, x.value, y.value


class TestProblem2:
    def __init__(self, na: int = 100, La: float = 1000, nb: int = 100, Lb: float = 20):
        self.x_dim, self.La = na, La
        A = get_matrix(self.x_dim, self.x_dim, np.linspace(1e-2, self.La, self.x_dim))
        self.A = A.T @ A
        self.a = np.random.random(self.x_dim)

        self.y_dim, self.Lb = nb, Lb
        B = get_matrix(self.y_dim, self.y_dim, np.linspace(1e-2, self.Lb, self.y_dim))
        self.B = B.T @ B
        self.b = np.random.random(self.y_dim)

        # находим решение получившейся задачи
        self.x1_star = - np.linalg.pinv(self.A) @ self.a
        self.x2_star = - np.linalg.pinv(self.B) @ self.b
        self.f_star = self.calc(self.x1_star, self.x2_star)[0]


    def calc(self, x1, x2):
        res =  0.5 * np.transpose(x1) @ self.A @ x1
        res += np.transpose(self.a) @ x1
        res += 0.5 * np.transpose(x2) @ self.B @ x2
        res += np.transpose(self.b) @ x2

        grad_x1 = self.A @ x1 + self.a
        grad_x2 = self.B @ x2 + self.b

        return res, grad_x1, grad_x2
