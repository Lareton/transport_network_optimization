import networkx as nx
import graph_tool as gt
from graph_tool.topology import shortest_distance
import numpy as np
import typing

import numba
from numba.core import types


def edge_dict_to_arr(d: dict, edge_to_ind: dict) -> np.ndarray:
    arr = np.zeros(len(d))
    for edge, value in d.items():
        arr[edge_to_ind[edge]] = value
    return arr


@numba.njit
def sum_flows_from_tree(source: int, targets: np.ndarray, pred_map_arr: np.ndarray, d_ij: np.ndarray, 
                        edge_to_ind: numba.typed.Dict) -> np.ndarray:
    num_edges = len(edge_to_ind) 
    flows_je = np.zeros((targets.size, num_edges))
    for v in targets:
        corr = d_ij[source, v]
        while v != source:
            v_pred = pred_map_arr[v]
            flows_je[v, edge_to_ind[(v_pred, v)]] += corr 
            v = v_pred
    return flows_je


def get_graphtool_graph(nx_graph: nx.Graph) -> gt.Graph:
    """Creates `gt_graph: graph_tool.Graph` from `nx_graph: nx.Graph`.
    Nodes in `gt_graph` are labeled by their indices in `nx_graph.edges()` instead of their labels
    (possibly of `str` type) in `nx_graph`"""
    nx_edges = nx_graph.edges()
    nx_edge_to_ind = dict(zip(nx_edges, list(range(len(nx_graph.edges())))))

    nx_bandwidths = edge_dict_to_arr(nx.get_edge_attributes(nx_graph, "bandwidth"), nx_edge_to_ind)
    nx_costs = edge_dict_to_arr(nx.get_edge_attributes(nx_graph, "cost"), nx_edge_to_ind)

    nx_nodes = list(nx_graph.nodes())
    edge_list = []
    for i, e in enumerate(nx_graph.edges()):
        edge_list.append((*[nx_nodes.index(v) for v in e],  nx_bandwidths[i], nx_costs[i]))

    gt_graph = gt.Graph(edge_list, eprops=[("bandwidths", "double"), ("costs", "double")])

    return gt_graph


class CapacityDualModel:
    def __init__(self, nx_graph: nx.Graph, d_ij: np.ndarray):
        self.graph = get_graphtool_graph(nx_graph)
        self.d_ij = d_ij

    def primal(self, flows_e: np.ndarray) -> float:
        return self.graph.ep.costs.a @ flows_e
    
    def dual(self, dual_costs: np.ndarray, flows_subgd_e: np.ndarray) -> float:
        return (self.graph.ep.costs.a + dual_costs) @ flows_subgd_e - dual_costs @ self.graph.ep.bandwidths.a
    
    def constraints_violation_l1(self, flows_e: np.ndarray) -> float:
        return np.maximum(0, flows_e - self.graph.ep.bandwidths.a).sum()

    def dual_subgradient(self, flows_subgd_e: np.ndarray) -> np.ndarray:
        return flows_subgd_e - self.graph.ep.bandwidths.a

    def flows_on_shortest(self, dual_costs: np.ndarray) -> np.ndarray:
        """Returns flows on edges for each ij-pair
        (obtained from flows on shortest paths w.r.t costs induced by dual_costs)"""
        num_nodes, num_edges = self.graph.num_vertices(), self.graph.num_edges()

        weights = self.graph.new_edge_property("double")
        weights.a = self.graph.ep.costs.a + dual_costs

        edges_arr = self.graph.get_edges()
        edge_to_ind = numba.typed.Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=numba.core.types.int64)
        for i, edge in enumerate(edges_arr):
            edge_to_ind[tuple(edge)] = i
    
        flows_on_shortest_ije = np.zeros((num_nodes, num_nodes, num_edges))
        targets = np.arange(num_nodes)
        for source in range(num_nodes):
            _, pred_map = shortest_distance(self.graph, source=source, target=targets, weights=weights, pred_map=True)
            flows_on_shortest_ije[source, :, :] = sum_flows_from_tree(
                source=source,
                targets=targets,
                pred_map_arr=np.array(pred_map.a),
                d_ij=self.d_ij,
                edge_to_ind=edge_to_ind,
            )

        return flows_on_shortest_ije


def subgd_mincost_mcf(
    model: CapacityDualModel,
    R: float,
    eps_abs: float,
    eps_cons_abs: float,
    max_iter: int = 10000,
) -> tuple:
    num_nodes, num_edges = model.graph.num_vertices(), model.graph.num_edges()
    flows_averaged_ije = np.zeros((num_nodes, num_nodes, num_edges))

    dual_costs = np.zeros(num_edges)

    dgap_log = []
    cons_log = []

    S = 0  # sum of stepsizes

    for k in range(max_iter):
        # inlined subgradient calculation with paths set saving
        flows_subgd_ije = model.flows_on_shortest(dual_costs)
        flows_subgd_e = flows_subgd_ije.sum(axis=(0, 1))
        subgd = -model.dual_subgradient(flows_subgd_e)  # grad of varphi = -dual

        h = R / (k + 1) ** 0.5 / np.linalg.norm(subgd)

        dual_val = model.dual(dual_costs, flows_subgd_e)

        flows_averaged_ije = (S * flows_averaged_ije + h * flows_subgd_ije) / (S + h)
        S += h

        flows_averaged_e = flows_averaged_ije.sum(axis=(0, 1))

        dgap_log.append(model.primal(flows_averaged_e) - dual_val)
        cons_log.append(model.constraints_violation_l1(flows_averaged_e))

        if dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
            break

        dual_costs = np.maximum(0, dual_costs - h * subgd)

    return dual_costs, flows_averaged_ije, dgap_log, cons_log


def ustm_mincost_mcf(
    model: CapacityDualModel,
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
    
    def func_grad_flows(dual_costs: np.ndarray):
        """func = varphi = -dual"""
        flows_subgd_ije = model.flows_on_shortest(dual_costs)
        flows_subgd_e = flows_subgd_ije.sum(axis=(0,    1))
        dual_grad = model.dual_subgradient(flows_subgd_e)
        return -model.dual(dual_costs, flows_subgd_e), -dual_grad, flows_subgd_ije

        # model.dual -  двойственная функция
        # dual_grad - градиент (по лямда, мю, по т) F
        # flows_subgd_ije - потоки при данных t
        # допом потом возвращать корреспонденции

    _, grad_y, flows_averaged_ije = func_grad_flows(y_start)
    L_value = np.linalg.norm(grad_y) / 10
    
    A = u = t = y = None
    inner_iters_num = 0

    for k in range(max_iter):
        while True:
            inner_iters_num += 1
    
            alpha = 0.5 / L_value + (0.25 / L_value**2 + A_prev / L_value) ** 0.5
            A = A_prev + alpha
    
            y = (alpha * u_prev + A_prev * t_prev) / A
            func_y, grad_y, flows_y = func_grad_flows(y)
            grad_sum = grad_sum_prev + alpha * grad_y
            
            u = np.maximum(0, y_start - grad_sum)
            
            t = (alpha * u + A_prev * t_prev) / A
            func_t, _, _ = func_grad_flows(t)
            
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
        
        gamma = alpha / A
        flows_averaged_ije = flows_averaged_ije * (1 - gamma) + flows_y * gamma
        flows_averaged_e = flows_averaged_ije.sum(axis=(0, 1))

        dgap_log.append(model.primal(flows_averaged_e) + func_t)
        cons_log.append(model.constraints_violation_l1(flows_averaged_e))
        A_log.append(A)
    
        if stop_by_crit and dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
            break

    return t, flows_averaged_ije, dgap_log, cons_log, A_log
