from numba.experimental import jitclass


@jitclass()
class JitTransportGraph:

    def __init__(self, nodes_number, links_number, max_path_length,
                 sources, targets, capacities, free_flow_times,
                 in_pointers, in_edges_array, pred,
                 out_pointers, out_edges_array, succ):
        self._nodes_number = nodes_number
        self._links_number = links_number
        self._max_path_length = max_path_length

        self._capacities = capacities
        self._free_flow_times = free_flow_times
        self._sources = sources
        self._targets = targets

        self._in_pointers = in_pointers
        self._in_edges_array = in_edges_array
        self._pred = pred
        self._out_pointers = out_pointers
        self._out_edges_array = out_edges_array
        self._succ = succ

    @property
    def nodes_number(self):
        return self._nodes_number

    @property
    def links_number(self):
        return self._links_number

    @property
    def max_path_length(self):
        return self._max_path_length

    @property
    def capacities(self):
        # return np.array(self.graph_table[['Capacity']]).flatten()
        return self._capacities

    @property
    def free_flow_times(self):
        # return np.array(self.graph_table[['Free Flow Time']]).flatten()
        return self._free_flow_times

    def successors(self, node_index):
        # return list(self.transport_graph.successors(vertex))
        return self._succ[self._out_pointers[node_index]: self._out_pointers[node_index + 1]]

    def predecessors(self, node_index):
        # return list(self.transport_graph.predecessors(vertex))
        return self._pred[self._in_pointers[node_index]: self._in_pointers[node_index + 1]]

    def in_edges(self, node_index):
        # return self._edges_indices(self.transport_graph.in_edges(vertex, data = True))
        return self._in_edges_array[self._in_pointers[node_index]: self._in_pointers[node_index + 1]]

    def out_edges(self, node_index):
        # return self._edges_indices(self.transport_graph.out_edges(vertex, data = True))
        return self._out_edges_array[self._out_pointers[node_index]: self._out_pointers[node_index + 1]]

    def source_of_edge(self, edge_index):
        # return self.graph_table.get_value(edge_index, 0, takeable=True)
        return self._sources[edge_index]

    def target_of_edge(self, edge_index):
        # return self.graph_table.get_value(edge_index, 1, takeable=True)
        return self._targets[edge_index]
