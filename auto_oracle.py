import numpy as np
from numba import int64
from numba import jit
from numba.experimental import jitclass


@jit(["float64(float64[:])"])
def logsumexp(ns):
    _max = np.max(ns)
    if np.isinf(_max):
        return - np.inf
    ds = ns - _max
    exp_sum = np.exp(ds).sum()
    return _max + np.log(exp_sum)


@jitclass()
class JitAutomaticOracle:
    """
    Oracle for automatic calculations of function kGamma * \Psi (t)
    """

    def __init__(self, graph, source, corr_targets, corr_values, gamma):
        # stock graph
        self.graph = graph
        self.nodes_number = graph.nodes_number
        self.edges_number = graph.links_number
        self.path_max_length = graph.max_path_length

        self.source = source
        self.corr_targets = corr_targets
        self.corr_values = corr_values
        self.targets = np.zeros(self.edges_number, dtype=int64)
        for edge in range(self.edges_number):
            self.targets[edge] = graph.target_of_edge(edge)
        self.gamma = gamma

        self.t_current = np.zeros(self.edges_number)
        self.A_values = np.empty((self.path_max_length + 1, self.nodes_number))
        self.B_values = np.empty((self.path_max_length + 1, self.nodes_number))

    def func(self, t_parameter):
        # print('automatic func called...')
        self.t_current[:] = t_parameter
        self._calculate_a_b_values()

        return np.dot(self.corr_values,
                      self.B_values[self.path_max_length][self.corr_targets])

    def grad(self, t_parameter):
        # assert(np.all(self.t_current == t_parameter))
        # print('automatic grad called...')
        gradient_vector = np.zeros(self.edges_number)

        # psi_d_beta_values initial values path_length = kMaxPathLength
        psi_d_beta_values = np.zeros(self.nodes_number)
        psi_d_beta_values[self.corr_targets] = self.corr_values
        psi_d_alpha_values = np.zeros(self.nodes_number)
        alpha_d_time = self._alpha_d_time_function(self.path_max_length)

        for path_length in range(self.path_max_length - 1, 0, -1):
            beta_d_beta_values = self._beta_d_beta_function(path_length + 1)
            psi_d_beta_values[:] = psi_d_beta_values * beta_d_beta_values

            # calculating psi_d_alpha_values
            beta_d_alpha_values = self._beta_d_alpha_function(path_length)

            psi_d_alpha_values = psi_d_beta_values * beta_d_alpha_values - \
                                 np.array([np.dot(psi_d_alpha_values[self.graph.successors(node)],
                                                  alpha_d_time[self.graph.out_edges(node)]) for
                                           node in range(self.nodes_number)])

            # calculating gradient
            alpha_d_time = self._alpha_d_time_function(path_length)
            gradient_vector += psi_d_alpha_values[self.targets] * alpha_d_time
        # print('my result = ' + str(gradient_vector))
        return gradient_vector

    def _alpha_d_time_function(self, path_length):
        # print('alpha_d_time_func called...')
        result = np.zeros(self.edges_number)
        if path_length == 1:
            result[self.graph.out_edges(self.source)] = - 1.0
        else:
            for node in range(self.nodes_number):
                A_node = self.A_values[path_length][node]
                if not np.isinf(A_node):
                    A_source = self.A_values[path_length - 1][self.graph.predecessors(node)]
                    in_edges = self.graph.in_edges(node)
                    result[in_edges] = - np.exp((A_source - self.t_current[in_edges] - A_node) /
                                                self.gamma)
        return result

    def _beta_d_beta_function(self, path_length):
        if path_length == 1:
            return np.zeros(self.nodes_number)
        beta_new = self.B_values[path_length][:]
        beta_old = self.B_values[path_length - 1][:]

        indices = np.nonzero(np.logical_not(np.isinf(beta_new)))
        result = np.zeros(self.nodes_number)
        result[indices] = np.exp((beta_old[indices] - beta_new[indices]) / self.gamma)
        return result

    def _beta_d_alpha_function(self, path_length):
        if path_length == 1:
            return np.ones(self.nodes_number)
        alpha_values = self.A_values[path_length][:]
        beta_values = self.B_values[path_length][:]

        indices = np.nonzero(np.logical_not(np.isinf(beta_values)))
        result = np.zeros(self.nodes_number)
        result[indices] = np.exp((alpha_values[indices] - beta_values[indices]) / self.gamma)
        return result

    def _calculate_a_b_values(self):
        self.A_values = np.full(self.A_values.shape, - np.inf)
        self.B_values = np.full(self.B_values.shape, - np.inf)
        initial_values = - 1.0 * self.t_current[self.graph.out_edges(self.source)]
        self.A_values[1][self.graph.successors(self.source)] = initial_values
        self.B_values[1][self.graph.successors(self.source)] = initial_values

        for path_length in range(2, self.path_max_length + 1):
            for term_vertex in range(self.nodes_number):
                if len(self.graph.predecessors(term_vertex)) > 0:
                    alpha = self.gamma * logsumexp(1.0 / self.gamma *
                                                   (self.A_values[path_length - 1][self.graph.predecessors(term_vertex)]
                                                    - self.t_current[self.graph.in_edges(term_vertex)]))

                    beta = self.gamma * logsumexp(np.array([1.0 / self.gamma *
                                                            self.B_values[path_length - 1][term_vertex],
                                                            1.0 / self.gamma * alpha]))

                    self.A_values[path_length][term_vertex] = alpha
                    self.B_values[path_length][term_vertex] = beta
