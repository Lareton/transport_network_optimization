import os
import pandas as pd
from scanf import scanf
import numpy as np
import re

def get_network_df(net_fname: str) -> pd.DataFrame:
    metadata = ''
    with open(net_fname, 'r') as fp:
        for index, line in enumerate(fp):
            if re.search(r'^~', line) is not None:
                skip_lines = index + 1
                headlist = re.findall(r'[\w]+', line)
                break
            else:
                metadata += line
                
    
    net_df = pd.read_csv(net_fname, skiprows=8, sep='\t')
    
    trimmed = [s.strip().lower() for s in net_df.columns]
    net_df.columns = trimmed
    
    # And drop the silly first and last columns
    net_df.drop(['~', ';'], axis=1, inplace=True)
    
    net_df = net_df[["init_node", "term_node", "capacity", "free_flow_time"]]
    
    # -1 because indices in the input data start from 1
    net_df.init_node -= 1
    net_df.term_node -= 1
    
    return net_df


def get_corrs(corrs_fname: str) -> np.ndarray:
    with open(corrs_fname, 'r') as myfile:
        trips_data = myfile.read()

    total_od_flow = scanf('<TOTAL OD FLOW> %f', trips_data)[0]
    zones_num = scanf('<NUMBER OF ZONES> %d', trips_data)[0]

    origins_data = re.findall(r'Origin[\s\d.:;]+', trips_data)

    corrs = np.zeros((zones_num, zones_num))
    for data in origins_data:
        origin = scanf('Origin %d', data)[0]
        origin_correspondences = re.findall(r'[\d]+\s+:[\d.\s]+;', data)
        targets = []
        corrs_from_zone = []
        for line in origin_correspondences:
            target, corr = scanf('%d : %f', line)
            targets.append(target)
            corrs_from_zone.append(corr)
        # -1 because indices in the input data start from 1
        targets = np.array(targets) - 1
        corrs[origin - 1, targets] = corrs_from_zone

    return corrs

