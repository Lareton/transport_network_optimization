import numba
import numpy as np
from dataclasses import dataclass
from typing import TypeVar

import scipy
from graph_tool.topology import shortest_distance

from numba import njit
from numba.core import types


@dataclass
class HyperParams:
    gamma: float
    mu_pow: float
    rho: float


@dataclass
class OptimParams:
    t: np.ndarray
    la: np.ndarray
    mu: np.ndarray


class DualOracle:

    def __init__(self, graph, net_df, l, w, params):
        self.params = params
        self.graph = graph
        self.net_df = net_df
        edges_arr = graph.get_edges()
        self.edge_to_ind = numba.typed.Dict.empty(key_type=types.UniTuple(types.int64, 2),
                                                  value_type=numba.core.types.int64)
        for i, edge in enumerate(edges_arr):
            self.edge_to_ind[tuple(edge)] = i

        self.zones_num = len(l)
        self.edges_num = len(graph.ep.capacity.a)
        self.nodes_cnt = self.zones_num
        self.edge_cnt = self.edges_num

        self.l = l
        self.w = w

        assert len(l) == len(w)

        self.f_bar = np.array(net_df['capacity'])
        self.t_bar = np.array(net_df['free_flow_time'])

    #         self.t = self.t_bar.copy() + np.random.rand(self.edge_cnt)
    #         self.la = np.zeros(self.zones_num)
    #         self.mu = snp.zeros(self.zones_num)

    #     @njit
    def sum_flows_from_tree(self, flows, source, targets, pred_map_arr, d):
        for v in targets:
            corr = d[source, v]
            while v != source:
                v_pred = pred_map_arr[v]
                # print(v, source, pred_map_arr)
                flows[self.edge_to_ind[(v_pred, v)]] += corr
                v = v_pred
        return flows

    def calc_F(self, optim_params, T):
        logsum_term = self.params.gamma * scipy.special.logsumexp(
            (-T + optim_params.la[..., None] + optim_params.mu[None, ...]) / self.params.gamma)
        return logsum_term - self.l @ optim_params.la - self.w @ optim_params.mu + np.sum(self.sigma_star(optim_params))

    def get_d(self, optim_params, T):
        exp_arg = (-T + optim_params.la[..., None] + optim_params.mu) / self.params.gamma
        exp_arg -= exp_arg.max()
        exps = np.exp(exp_arg)
        return exps / exps.sum()

    def invert_tau(self, optim_params):
        return self.f_bar * ((optim_params.t ** self.params.mu_pow - self.t_bar ** self.params.mu_pow) / (
                    self.params.rho * self.t_bar ** self.params.mu_pow))

    def grad_dF_dt(self, optim_params, flows_on_shortest):
        return -flows_on_shortest + self.invert_tau(optim_params)

    def grad_dF_dla(self, d):
        val = d.sum(axis=1) - self.l
        print("grad dF dla: ", np.linalg.norm(val))
        return val

    def grad_dF_dmu(self, d):
        val = d.sum(axis=0) - self.w
        print("grad dF dmu: ", np.linalg.norm(val))
        return val

    def get_flows_on_shortest(self, sources, targets, d, pred_maps):
        flows_on_shortest = np.zeros(self.edges_num)
        for ind, source in enumerate(sources):
            pred_map = pred_maps[ind]
            self.sum_flows_from_tree(flows_on_shortest, source, targets, np.array(pred_map.a), d)
        return flows_on_shortest

    def get_T_and_predmaps(self, g, optim_params, sources, targets):
        T = np.zeros((len(sources), len(targets)))
        pred_maps = []

        for source in sources:
            # CHECK ME - правильно ли сопоставляются веса ребер (от инициализации)
            weights = g.ep.fft
            weights.a = optim_params.t

            short_distances, pred_map = shortest_distance(g, source=source, target=targets, weights=weights,
                                                          pred_map=True)
            #             print("short_distances sum: ", short_distances.sum())
            pred_maps.append(pred_map)

            for j in range(len(short_distances)):
                T[source, j] = short_distances[j]

        return T, pred_maps

    def sigma_star(self, optim_params):
        # return self.f_bar * ((t - t_bar) / (t_bar * self.params.rho)) ** self.params.mu_pow * \
        #        (t - t_bar) / (1 + self.params.mu_pow)
        return self.invert_tau(optim_params) * (optim_params.t - self.t_bar) / (1 + self.params.mu_pow)

    def sigma(self, f):
        return self.t_bar * f * (1 + (self.params.rho / (1 + 1 / self.params.mu_pow)) *
                                 (f / self.f_bar) ** (1 / self.params.mu_pow))

    def prime(self, f, d):
        return np.sum(self.sigma(f)) + self.params.gamma + np.sum(d * np.log(d))

