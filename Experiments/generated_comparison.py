import sys
import os
sys.path.append(r"C:\Users\lacke\Desktop\github\Simplicial_Ratio\Experiments")
import dataset_comparison as dc
import numpy as np
import random
from math import comb
from itertools import combinations as combs
import time
from collections import Counter
import xgi
import hypercontagion as hc
import matplotlib.pyplot as plt
import time
import random
import Hypergraph_Models as hm


datasets = [
    "contact-primary-school",
    "contact-high-school",
    "hospital-lyon",
    "email-enron",
    "email-eu",
    "ndc-substances",
    "diseasome",
    "disgenenet",
    "congress-bills",
    "tags-ask-ubuntu",
]

def edge_size_distribution(H):
    sizes = [len(H.edges.members(e)) for e in H.edges]
    counts = Counter(sizes)
    max_size = max(counts)
    distribution = [counts.get(i, 0) for i in range(max_size + 1)]
    return distribution

if __name__ == "__main__":
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    dataset = datasets[6]
    H = xgi.load_xgi_data(dataset)
    num_nodes = H.num_nodes
    edges = edge_size_distribution(H)
    degree_seq = [H.degree(n) for n in H.nodes]

    for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:      
        idx = int(q * 10 - 1) 
        dc.run_multiple_SIR_with_errorbands(
            num_nodes, 
            edges, 
            degree_seq,
            gamma = 0.05,
            colors = ["#00B388","#DA291C", "#418FDF"],
            num_graphs = 10,
            ax = axes[idx])
        axes[idx].set_title(f"q = {q}", fontsize=12)
    
    fig.suptitle("SIR on Synthetic Data with Error Bands", fontsize=16) 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Save the figure
    fig.savefig("SIR on Synthetic Data with Error Bands.png", dpi=300, bbox_inches='tight')
    
