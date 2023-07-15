import networkx as nx
import graph_tool as gt
from graph_tool.topology import shortest_distance
import numpy as np
import typing
from dataclasses import dataclass, field

import numba
from numba.core import types
from tqdm import tqdm
from typing import List, Tuple

from transport_problem import OptimParams, DualOracle, HyperParams



@dataclass
class USTM_Results:
    history_la_grad_norm: List[float] = field(default_factory=list)
    history_mu_grad_norm: List[float] = field(default_factory=list)
    history_dual_values: List[float] = field(default_factory=list)
    history_prime_values: List[float] = field(default_factory=list)
    history_dual_gap: List[float] = field(default_factory=list)
    history_A: List[float] = field(default_factory=list)
    history_la_mu_grad_norm: List[float] = field(default_factory=list)
    history_count_calls: List[int] = field(default_factory=list)
    custom_critery: List[int] = field(default_factory=list)

    d_avaraged: np.ndarray = None
    flows_averaged: np.ndarray = None
    t_avaraged: np.ndarray = None
    count_oracle_calls: int = 0


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
        # print("vars block grad: ", np.linalg.norm(vars_block))
        assert len(vars_block) == self.T_LEN + self.LA_LEN + self.MU_LEN
        self.optim_params.t = vars_block[:self.T_LEN]
        self.optim_params.t = np.maximum(self.optim_params.t, self.oracle.t_bar)

        # print("t in optim params grad: ", np.linalg.norm(self.optim_params.t), np.linalg.norm(vars_block[:self.T_LEN]))
        # print("la in optim params norm: ", np.linalg.norm(self.optim_params.la))
        # print("mu in optim params norm: ", np.linalg.norm(self.optim_params.mu))

        self.optim_params.la = vars_block[self.T_LEN:self.T_LEN + self.LA_LEN]
        self.optim_params.mu = vars_block[self.T_LEN + self.LA_LEN:]

        # T, pred_maps = self.oracle.get_T_and_predmaps_parallel(self.optim_params, self.sources, self.targets)
        T, pred_maps = self.oracle.get_T_and_predmaps(self.optim_params, self.sources, self.targets)

        self.d = self.oracle.get_d(self.optim_params, T)
        flows_on_shortest = self.oracle.get_flows_on_shortest(self.sources, self.targets, self.d, pred_maps)

        grad_t = self.oracle.grad_dF_dt(self.optim_params, flows_on_shortest)
        grad_la = self.oracle.grad_dF_dla(self.d)
        grad_mu = self.oracle.grad_dF_dmu(self.d)

        full_grad = np.hstack([grad_t, grad_la, grad_mu])
        dual_value = self.oracle.calc_F(self.optim_params, T)

        self.flows = flows_on_shortest.copy()  # self.oracle.get_flows_on_shortest(self.sources, self.targets, self.d, pred_maps)

        return dual_value, flows_on_shortest, full_grad, grad_t, grad_la, grad_mu

    def get_prime_value(self):
        return self.oracle.prime(self.flows, self.d)

    def get_init_vars_block(self):
        return np.hstack([self.oracle.t_bar.copy(), np.zeros(self.LA_LEN), np.zeros(self.MU_LEN)])


# TODO: убрать unused переменные
def ustm_mincost_mcf(
        oracle_stacker: OracleStacker,
        eps_abs: float,
        eps_cons_abs: float,
        max_iter: int = 10000,
        stop_by_crit: bool = True,
) -> USTM_Results:
    results = USTM_Results()

    A_prev = 0.0

    # t_start = np.zeros(oracle_stacker.parameters_vector_size)  # dual costs w
    t_start = oracle_stacker.get_init_vars_block()  # dual costs w

    y_start = u_prev = t_prev = np.copy(t_start)
    assert y_start is u_prev  # acceptable at first initialization

    grad_sum_prev = np.zeros(len(t_start))

    print("y_start: ", y_start)
    func_t, flows_averaged, grad_y, grad_t, grad_la, grad_mu = oracle_stacker(y_start)
    print("first exceeding the limits: ", np.linalg.norm(np.hstack([grad_la, grad_mu])))
    results.count_oracle_calls += 1

    d_avaraged = oracle_stacker.d.copy()

    L_value = np.linalg.norm(grad_y) / 10

    A = u = t = y = None
    inner_iters_num = 0

    print("start optimizing")

    results.history_count_calls.append(results.count_oracle_calls)
    results.history_dual_values.append(func_t)
    results.history_prime_values.append(oracle_stacker.oracle.prime(flows_averaged, d_avaraged))
    norm_la_mu_grad = np.linalg.norm(np.hstack([grad_la, grad_mu]))
    results.history_la_mu_grad_norm.append(norm_la_mu_grad)
    results.custom_critery.append(min(abs(func_t), norm_la_mu_grad))
    results.history_dual_gap.append(oracle_stacker.oracle.prime(flows_averaged, d_avaraged) + func_t)
    results.history_A.append(0)

    print("first dual_func: ", oracle_stacker.oracle.prime(flows_averaged, d_avaraged) + func_t )

    # for k in tqdm(range(max_iter)):
    for k in tqdm(range(max_iter)):
        while True:
            inner_iters_num += 1

            alpha = 0.5 / L_value + (0.25 / L_value ** 2 + A_prev / L_value) ** 0.5
            A = A_prev + alpha

            y = (alpha * u_prev + A_prev * t_prev) / A
            # y[:oracle_stacker.T_LEN] = np.maximum(oracle_stacker.oracle.t_bar, y[:oracle_stacker.T_LEN]) # FIXME ???

            func_y, flows_y, grad_y, *_ = oracle_stacker(y)
            results.count_oracle_calls += 1

            grad_sum = grad_sum_prev + alpha * grad_y

            u = y_start - grad_sum

            # print("count values below t_bar in old t: ", (u[:oracle_stacker.T_LEN] < oracle_stacker.oracle.t_bar).sum())
            u[:oracle_stacker.T_LEN] = np.maximum(oracle_stacker.oracle.t_bar, u[:oracle_stacker.T_LEN])
            # print("count values below t_bar in new t: ", (u[:oracle_stacker.T_LEN] < oracle_stacker.oracle.t_bar).sum())

            # u = np.maximum(0, y_start - grad_sum)

            t = (alpha * u + A_prev * t_prev) / A

            func_t, _, full_grad, grad_t, grad_la, grad_mu = oracle_stacker(t)
            results.count_oracle_calls += 1

            lvalue = func_t

            # print("norm (t - y): ", np.linalg.norm(t - y))
            # print("norm t: ", np.linalg.norm(oracle_stacker.optim_params.t))
            # print("norm la: ", np.linalg.norm(oracle_stacker.optim_params.la))
            # print("norm mu: ", np.linalg.norm(oracle_stacker.optim_params.mu))
            # print()

            rvalue = (func_y + np.dot(grad_y, t - y) + 0.5 * L_value * np.sum((t - y) ** 2) +
                      #                      0.5 * alpha / A * eps_abs )  # because, in theory, noise accumulates
                      0.5 * eps_abs)

            if lvalue <= rvalue:
                break
            else:
                L_value *= 2

            assert L_value < np.inf

        # history_dual_values.append(func_y)
        #         history_prime_values.append(oracle_stacker.get_prime_value())

        # cnt_oracle_calls = results.count_oracle_calls
        results.history_count_calls.append(results.count_oracle_calls)
        results.history_dual_values.append(func_t)
        results.history_prime_values.append(oracle_stacker.oracle.prime(flows_averaged, d_avaraged))
        norm_la_mu_grad = np.linalg.norm(np.hstack([grad_la, grad_mu]))
        results.history_la_mu_grad_norm.append(norm_la_mu_grad)
        results.custom_critery.append(min(abs(func_t), norm_la_mu_grad))

        A_prev = A
        L_value /= 2

        t_prev = t
        u_prev = u
        grad_sum_prev = grad_sum

        teta = alpha / A
        flows_averaged = flows_averaged * (1 - teta) + flows_y * teta
        d_avaraged = d_avaraged * (1 - teta) + oracle_stacker.d * teta

        results.d_avaraged = oracle_stacker.d

        results.history_dual_gap.append(oracle_stacker.oracle.prime(flows_averaged, d_avaraged) + func_t)
        results.history_A.append(A)

        if stop_by_crit and abs(results.history_dual_gap[-1]) <= eps_abs and results.history_la_mu_grad_norm[-1] <= eps_cons_abs:
            print("EARLY STOPPING")
            break

    return results
