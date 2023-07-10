import networkx as nx
import graph_tool as gt
from graph_tool.topology import shortest_distance
import numpy as np
import typing

import numba
from numba.core import types

from transport_problem import OptimParams, DualOracle, HyperParams

T_LEN = 76
LA_LEN = 25
MU_LEN = 25

graph = None # TODO создавать граф
oracle = None # TODO - создать оракла
sources, targets = None, None    # определять sources и targets
oracle_stacker = OracleStacker(oracle, graph, sources, targets)


class OracleStacker:
    def __init__(self, oracle: DualOracle, graph, sources, targets):
        self.oracle = oracle
        self.graph = graph
        self.sources = sources
        self.targets = targets

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
        assert len(vars_block) == T_LEN + LA_LEN + MU_LEN
        t = vars_block[:T_LEN]
        la = vars_block[T_LEN:T_LEN + LA_LEN]
        t = vars_block[T_LEN + LA_LEN:]

        T, pred_maps = self.oracle.get_T_and_predmaps(self.graph, self.optim_params, self.sources, self.targets)
        d = self.oracle.get_d(self.optim_params, T)
        grad_t = self.oracle.get_flows_on_shortest(self.sources, self.targets, d, pred_maps)
        grad_la = self.oracle.grad_dF_dla(self.optim_params, T)
        grad_mu = self.oracle.grad_dF_dmu(self.optim_params, T)

        full_grad = np.stack([grad_t, grad_la, grad_mu], axis=0)
        dual_value = self.oracle.calc_F(self.optim_params, T)

        flows = self.oracle.get_flows_on_shortest(self.sources, self.targets, d, pred_maps)

        return dual_value, full_grad, flows



def ustm_mincost_mcf(
    model,
    eps_abs: float,
    eps_cons_abs: float,
    max_iter: int = 10000,
    stop_by_crit: bool = True,
) -> tuple:
    dgap_log = []
    cons_log = []
    A_log = []
    
    A_prev = 0.0
    t_start = np.zeros(model.graph.num_edges())  # dual costs w
    y_start = u_prev = t_prev = np.copy(t_start)
    assert y_start is u_prev  # acceptable at first initialization
    grad_sum_prev = np.zeros(len(t_start))

    _, grad_y, flows_averaged = oracle_stacker(y_start)
    L_value = np.linalg.norm(grad_y) / 10
    
    A = u = t = y = None
    inner_iters_num = 0

    for k in range(max_iter):
        while True:
            inner_iters_num += 1
    
            alpha = 0.5 / L_value + (0.25 / L_value**2 + A_prev / L_value) ** 0.5
            A = A_prev + alpha
    
            y = (alpha * u_prev + A_prev * t_prev) / A
            func_y, grad_y, flows_y = oracle_stacker(y)
            grad_sum = grad_sum_prev + alpha * grad_y
            
            u = np.maximum(0, y_start - grad_sum)
            
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
        flows_averaged = flows_averaged * (1 - teta) + flows_y * teta
        flows_averaged_e = flows_averaged.sum(axis=(0, 1))

        # dgap_log.append(model.primal(flows_averaged_e) + func_t)
        # cons_log.append(model.constraints_violation_l1(flows_averaged_e))
        A_log.append(A)
        # if stop_by_crit and dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
        #     break

    return t, flows_averaged, dgap_log, cons_log, A_log
