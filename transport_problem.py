import numba
import numpy as np
from dataclasses import dataclass
from typing import TypeVar

import scipy
from graph_tool.topology import shortest_distance

from numba import njit
from numba.core import types


@dataclass
class Params:
    gamma: float
    mu_pow: float
    rho: float


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
        self.t = self.t_bar.copy() + np.random.rand(self.edge_cnt)

        self.la = np.zeros(self.zones_num)
        self.mu = np.zeros(self.zones_num)

    @njit
    def sum_flows_from_tree(self, flows, source, targets, pred_map_arr, d, edge_to_ind):
        for v in targets:
            corr = d[source, v]
            while v != source:
                v_pred = pred_map_arr[v]
                flows[edge_to_ind[(v_pred, v)]] += corr
                v = v_pred
        return flows

    def calc_F(self, T):
        return self.params.gamma * scipy.special.logsumexp(
            (-T + self.la[..., None] + self.mu[
                None, ...]) / self.params.gamma) - self.params.l @ self.la - self.params.w @ self.mu + np.sum(
            self.sigma_star(self.t, self.t_bar, self.params.mu_pow, self.params.rho))

    def get_d(self, T):
        return np.exp((-T + self.la + self.mu) / self.params.gamma) / np.sum(
            np.exp((-T + self.la + self.mu) / self.params.gamma))

    def invert_tau(self, t):
        return self.f_bar * ((t - self.t_bar) / (self.params.k * self.t_bar)) ** self.params.mu_pow

    def grad_dF_dt(self, grad_t):
        return grad_t + self.invert_tau(self.t)

    def grad_dF_dla(self, T):
        return np.sum(np.exp(-T + self.la[..., None] + self.mu[None, ...]), axis=1) / np.sum(
            np.exp(-T + self.la[..., None] + self.mu[None, ...])) - self.params.l

    def grad_dF_dmu(self, T):
        return np.sum(np.exp(-T + self.la[..., None] + self.mu[None, ...]), axis=0) / np.sum(
            np.exp(-T + self.la[..., None] + self.mu[None, ...])) - self.params.w


    def get_grad_t(self, source, targets, d, pred_map):
        flows_on_shortest = np.zeros(self.edges_num)
        self.sum_flows_from_tree(flows_on_shortest, source, targets, np.array(pred_map.a), d,
                                     self.edge_to_ind)
        return flows_on_shortest
        
        

    def get_T_and_predmap(self, g, sources, targets):
        T = np.zeros((len(sources), len(targets)))

        for source in sources:
            short_distances, pred_map = shortest_distance(g, source=source, target=targets, weights=g.ep.fft,
                                                          pred_map=True)

            for j in range(len(short_distances)):
                T[source, j] = short_distances[j]

        return T, pred_map

    def sigma_star(self, t, t_bar, mu_pow, rho):
        return self.f_bar * ((t - t_bar) / (t_bar * rho)) ** mu_pow * (t - t_bar) / (1 + mu_pow)


def main():
    params = Params()


if __name__ == '__main__':
    main()
