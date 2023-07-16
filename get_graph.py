from pathlib import Path

from read_data import get_network_df, get_corrs, get_full_df


def get_graph_by_name(graph_type):
    """Загружает, предобрабатывает граф по названию. Возвращает net_df, corrs"""
    print("graph type: ", graph_type)
    # SMALL GRAPH:

    if graph_type == "SiouxFalls":
        net_df = get_network_df(Path('SiouxFalls') / 'SiouxFalls_net.tntp')
        corrs = get_corrs(Path('SiouxFalls') / 'SiouxFalls_trips.tntp')
        people_count = corrs.sum()
        corrs = corrs / people_count
        net_df.capacity /= people_count

    else:
        if  graph_type == "Berlin":
            BIG_CONST_FOR_FAKE_EDGES = 400
            net_df = get_network_df(Path('SiouxFalls') / 'berlin-center_net.tntp')
            corrs = get_corrs(Path('SiouxFalls') / 'berlin-center_trips.tntp')
        elif graph_type == "Chicago":
            BIG_CONST_FOR_FAKE_EDGES = 40
            net_df = get_network_df(Path('SiouxFalls') / 'ChicagoSketch_net.tntp')
            corrs = get_corrs(Path('SiouxFalls') / 'ChicagoSketch_trips.tntp')
        elif graph_type == "Anaheim":
            BIG_CONST_FOR_FAKE_EDGES = 4
            net_df = get_network_df(Path('SiouxFalls') / 'Anaheim_net.tntp')
            corrs = get_corrs(Path('SiouxFalls') / 'Anaheim_trips.tntp')

        print(net_df.free_flow_time.max())

        net_df.loc[net_df.free_flow_time < 1e-6, "free_flow_time"] = BIG_CONST_FOR_FAKE_EDGES
        people_count = corrs.sum()
        corrs = corrs / people_count
        net_df.capacity /= people_count
        net_df = net_df.rename({"free_flow_time": "fft"}, axis=1)

    return net_df, corrs
