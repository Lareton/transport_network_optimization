import networkx as nx
import graph_tool as gt
from graph_tool.topology import shortest_distance
import numpy as np
import typing

import numba
from numba.core import types
from tqdm import tqdm

from transport_problem import OptimParams, DualOracle, HyperParams


# graph = None # TODO создавать граф
# oracle = None # TODO - создать оракла
# sources, targets = None, None    # определять sources и targets
# oracle_stacker = OracleStacker(oracle, graph, sources, targets)


class OracleStacker:
    def __init__(self, oracle: DualOracle, graph, sources, targets):
        self.oracle = oracle
        self.graph = graph
        self.sources = sources
        self.targets = targets

        self.T_LEN = oracle.edges_num
        self.LA_LEN = oracle.zones_num
        self.MU_LEN = oracle.zones_num

        # размер вектора параметров [t, la, mu]
        self.parameters_vector_size = self.T_LEN + self.LA_LEN + self.MU_LEN

        self.t = oracle.t_bar.copy()
        self.la = np.zeros(oracle.zones_num)
        self.mu = np.zeros(oracle.zones_num)
        self.optim_params = OptimParams(self.t, self.la, self.mu)

    def __call__(self, vars_block, *args, **kwargs):
        """
        :param vars_block: все оптимизируемые переменные stack[t, la, mu]
        :return:
        dual_value -  значение двойстенной функции для t, la, mu
        full_grad - градиент, stack[t_grad, la_grad, mu_grad]
        flows_averaged -  потоки при данных t (f)
        """
        assert len(vars_block) == self.T_LEN + self.LA_LEN + self.MU_LEN
        self.optim_params.t = vars_block[:self.T_LEN]
        self.optim_params.la = vars_block[self.T_LEN:self.T_LEN + self.LA_LEN]
        self.optim_params.mu = vars_block[self.T_LEN + self.LA_LEN:]

        T, pred_maps = self.oracle.get_T_and_predmaps(self.graph, self.optim_params, self.sources, self.targets)
        d = self.oracle.get_d(self.optim_params, T)
        flows_on_shortest = self.oracle.get_flows_on_shortest(self.sources, self.targets, d, pred_maps)

        grad_t = self.oracle.grad_dF_dt(self.optim_params, flows_on_shortest)
        grad_la = self.oracle.grad_dF_dla(self.optim_params, T)
        grad_mu = self.oracle.grad_dF_dmu(self.optim_params, T)

        full_grad = np.hstack([grad_t, grad_la, grad_mu])
        dual_value = self.oracle.calc_F(self.optim_params, T)

        flows = self.oracle.get_flows_on_shortest(self.sources, self.targets, d, pred_maps)

        return dual_value, full_grad, flows


# TODO: убрать unused переменные
def ustm_mincost_mcf(
        oracle_stacker: OracleStacker,
        eps_abs: float,
        eps_cons_abs: float,
        max_iter: int = 10000,
        stop_by_crit: bool = True,
) -> tuple:
    dgap_log = []
    cons_log = []
    A_log = []

    A_prev = 0.0
    print(1)

    t_start = np.zeros(oracle_stacker.parameters_vector_size)  # dual costs w
    print(1)

    y_start = u_prev = t_prev = np.copy(t_start)
    assert y_start is u_prev  # acceptable at first initialization

    print(1)
    grad_sum_prev = np.zeros(len(t_start))

    _, grad_y, flows_averaged = oracle_stacker(y_start)
    L_value = np.linalg.norm(grad_y) / 10

    A = u = t = y = None
    inner_iters_num = 0

    print("start optimizing")
    for k in tqdm(range(max_iter)):
        while True:
            inner_iters_num += 1

            alpha = 0.5 / L_value + (0.25 / L_value ** 2 + A_prev / L_value) ** 0.5
            A = A_prev + alpha

            y = (alpha * u_prev + A_prev * t_prev) / A
            func_y, grad_y, flows_y = oracle_stacker(y)
            grad_sum = grad_sum_prev + alpha * grad_y

            tmp = np.maximum(oracle_stacker.oracle.t_bar, (y_start - grad_sum)[:oracle_stacker.T_LEN])
            u = y_start - grad_sum
            u[:oracle_stacker.T_LEN] = tmp
            # u = np.maximum(0, y_start - grad_sum)

            t = (alpha * u + A_prev * t_prev) / A
            func_t, _, _ = oracle_stacker(t)

            lvalue = func_t
            rvalue = (func_y + np.dot(grad_y, t - y) + 0.5 * L_value * np.sum((t - y) ** 2) +
                      #                      0.5 * alpha / A * eps_abs )  # because, in theory, noise accumulates
                      0.5 * eps_abs)

            if lvalue <= rvalue:
                break
            else:
                L_value *= 2

            assert L_value < np.inf

        A_prev = A
        L_value /= 2

        t_prev = t
        u_prev = u
        grad_sum_prev = grad_sum

        teta = alpha / A
        # TODO TODO
        # flows_averaged = flows_averaged * (1 - teta) + flows_y * teta
        # flows_averaged_e = flows_averaged.sum(axis=(0, 1))

        # dgap_log.append(model.primal(flows_averaged_e) + func_t)
        # cons_log.append(model.constraints_violation_l1(flows_averaged_e))
        A_log.append(A)
        # if stop_by_crit and dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
        #     break

    return t, flows_averaged, dgap_log, cons_log, A_log
