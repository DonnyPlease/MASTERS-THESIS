import numpy as np

def bins_from_bins_edges(bins_edges):
    return [(bins_edges[i] + bins_edges[i + 1]) / 2 for i in range(len(bins_edges) - 1)]