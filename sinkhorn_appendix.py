import networkx as nx
import graph_tool as gt
from graph_tool.topology import shortest_distance
import numpy as np
import typing

import numba
from numba.core import types
from tqdm import tqdm
from sinkhorn import Sinkhorn
from oracle_utils import AlgoResults

from transport_problem import OptimParams, DualOracle, HyperParams


class OracleSinkhornStacker:
    def __init__(self, oracle: DualOracle, graph, sources, targets, l, w, params):
        self.k = 0
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

        self.sinkhorn = Sinkhorn(l, w, max_iter=100000, eps=1e-3)
        self.params = params

    def __call__(self, vars_block, *args, **kwargs):
        global oracle_cnt
        """
        :param vars_block: все оптимизируемые переменные stack[t, la, mu]
        :return:
        dual_value -  значение двойстенной функции для t, la, mu
        full_grad - градиент, stack[t_grad, la_grad, mu_grad]
        flows_averaged -  потоки при данных t (f)
        """
        assert len(vars_block) == self.T_LEN

        self.optim_params.t = vars_block
        self.optim_params.t = np.maximum(self.optim_params.t, self.oracle.t_bar)

        T, pred_maps = self.oracle.get_T_and_predmaps_parallel(self.optim_params, self.sources, self.targets)
        self.k += 1

        self.d, self.optim_params.la, self.optim_params.mu, k = self.sinkhorn.run(T / self.params.gamma,
                                                                                  self.optim_params.la,
                                                                                  self.optim_params.mu)
        flows_on_shortest = self.oracle.get_flows_on_shortest(self.sources, self.targets, self.d, pred_maps)
        grad_t = self.oracle.grad_dF_dt(self.optim_params, flows_on_shortest)
        grad_la = self.oracle.grad_dF_dla(self.d)
        grad_mu = self.oracle.grad_dF_dmu(self.d)
        grad = np.linalg.norm(np.hstack([grad_la, grad_mu]))

        full_grad = grad_t
        dual_value = self.oracle.calc_F_via_d(self.optim_params, self.d, T)

        self.flows = self.oracle.get_flows_on_shortest(self.sources, self.targets, self.d, pred_maps)

        return dual_value, full_grad, flows_on_shortest, grad, k

    def get_prime_value(self):
        return self.oracle.prime(self.flows, self.d)

    def get_init_vars_block(self):
        return self.oracle.t_bar.copy()


# TODO: убрать unused переменные
def ustm_sinkhorn_mincost_mcf(
        oracle_stacker: OracleSinkhornStacker,
        eps_abs: float,
        crit_eps_abs: float,
        eps_cons_abs: float,
        max_iter: int = 10000,
        stop_by_crit: bool = True,
) -> tuple:
    history = AlgoResults()
    sinkhistory = AlgoResults()

    A_prev = 0.0

    t_start = oracle_stacker.get_init_vars_block()

    y_start = u_prev = t_prev = np.copy(t_start)
    assert y_start is u_prev

    grad_sum_prev = np.zeros(len(t_start))

    zero_dgap, grad_y, flows_averaged, grad, sinkhorn_cnt = oracle_stacker(y_start)
    sinkhorn_iters_cnt = sinkhorn_cnt
    oracle_calls = 1
    history.history_count_calls.append(oracle_calls)
    history.history_dual_gap.append(zero_dgap)
    history.history_la_mu_grad_norm.append(grad)
    sinkhistory.history_count_calls.append(sinkhorn_iters_cnt)
    sinkhistory.history_dual_gap.append(zero_dgap)
    sinkhistory.history_la_mu_grad_norm.append(grad)
    print(zero_dgap, grad)
    d_avaraged = oracle_stacker.d.copy()

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
            func_y, grad_y, flows_y, grad, sinkhorn_cnt = oracle_stacker(y)
            sinkhorn_iters_cnt += sinkhorn_cnt
            oracle_calls += 1

            grad_sum = grad_sum_prev + alpha * grad_y

            u = y_start - grad_sum
            u[:oracle_stacker.T_LEN] = np.maximum(oracle_stacker.oracle.t_bar, u[:oracle_stacker.T_LEN])

            t = (alpha * u + A_prev * t_prev) / A
            func_t, _, _, grad, sinkhorn_cnt = oracle_stacker(t)
            sinkhorn_iters_cnt += sinkhorn_cnt
            oracle_calls += 1

            lvalue = func_t

            rvalue = (func_y + np.dot(grad_y, t - y) + 0.5 * L_value * np.sum((t - y) ** 2) +
#                       0.5 * alpha / A * eps_abs )  # because, in theory, noise accumulates
                      0.5 * eps_abs)

            if lvalue <= rvalue:
                break
            else:
                L_value *= 2

            assert L_value < np.inf

        primal = oracle_stacker.oracle.prime(flows_averaged, d_avaraged)

        A_prev = A
        L_value /= 2

        t_prev = t
        u_prev = u
        grad_sum_prev = grad_sum

        teta = alpha / A
        flows_averaged = flows_averaged * (1 - teta) + flows_y * teta
        d_avaraged = d_avaraged * (1 - teta) + oracle_stacker.d * teta

        dgap = oracle_stacker.oracle.prime(flows_averaged, d_avaraged) + func_t
        history.history_count_calls.append(oracle_calls)
        history.history_dual_gap.append(dgap)
        history.history_la_mu_grad_norm.append(grad)
        sinkhistory.history_count_calls.append(sinkhorn_iters_cnt)
        sinkhistory.history_dual_gap.append(dgap)
        sinkhistory.history_la_mu_grad_norm.append(grad)

        if stop_by_crit and history.history_dual_gap[-1] <= crit_eps_abs and history.history_la_mu_grad_norm[
            -1] <= eps_cons_abs:
            print("STOP BY CRIT!!!")
            break

    return t, history, sinkhistory
