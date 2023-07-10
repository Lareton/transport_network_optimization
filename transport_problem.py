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
    l: np.array
    w: np.array
    f_flow: float
    mu_pow: float
    t_flow: float
    rho: float
    mu_pow: float


class DualOracle:

    def __init__(self, graph, net_df, corrs, la, mu, params):
        self.params = params
        self.graph = graph
        self.net_df = net_df
        self.corrs = corrs
        edges_arr = graph.get_edges()
        self.edge_to_ind = numba.typed.Dict.empty(key_type=types.UniTuple(types.int64, 2),
                                                  value_type=numba.core.types.int64)
        for i, edge in enumerate(edges_arr):
            self.edge_to_ind[tuple(edge)] = i

        self.zones_num = corrs.shape[0]
        self.edges_num = len(graph.ep.capacity.a)
        self.nodes_cnt = self.zones_num
        self.edge_cnt = self.edges_num

        self.f_bar = np.array(net_df['capacity'])
        self.t_bar = np.array(net_df['free_flow_time'])
        self.t = self.t_bar.copy() + np.random.rand(self.edge_cnt)

        self.la = la
        self.mu = mu

    @njit
    def sum_flows_from_tree(self, flows, source, targets, pred_map_arr, corrs, edge_to_ind):
        for v in targets:
            corr = corrs[source, v]
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

    def d(self, T):
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

    def get_T_and_grad_t(self, g, sources, targets):
        """
        Считает градиент F по t = суммарные потоки по ребра при прохождениии из всех истоков во все стоки
        """

        flows_on_shortest = np.zeros(self.edges_num)
        T = np.zeros((len(sources), len(targets)))

        for source in sources:
            short_distances, pred_map = shortest_distance(g, source=source, target=targets, weights=g.ep.fft,
                                                          pred_map=True)

            for j in range(len(short_distances)):
                T[source, j] = short_distances[j]

            self.sum_flows_from_tree(flows_on_shortest, source, targets, np.array(pred_map.a), self.corrs,
                                     self.edge_to_ind)

        return T, flows_on_shortest

    def sigma_star(self, t, t_bar, mu_pow, rho):
        return self.f_bar * ((t - t_bar) / (t_bar * rho)) ** mu_pow * (t - t_bar) / (1 + mu_pow)


def main():
    params = Params()


if __name__ == '__main__':
    main()
