
import sys
import numpy as np
from scipy.sparse.linalg import eigs
import xgi
import matplotlib.pyplot as plt
from collections import Counter
from termcolor import colored
import Hypergraph_Models as hm
from itertools import combinations as combs

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

# DB Matrix computation
def compute_DB_matrix(H):
    """
    Construct the DB matrix for a hypergraph H.
    Each hyperedge contributes weight 1/(|e|-1) between all its node pairs.
    """
    nodes = list(H.nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    DB = np.zeros((n, n))

    for e in H.edges.members():
        size = len(e)
        if size <= 1:
            continue
        weight = 1 / (size - 1)
        for i in e:
            for j in e:
                if i != j:
                    DB[node_index[i], node_index[j]] += weight
    return DB, nodes



# Compute critical infection rate
def compute_sir_threshold(DB, mu=0.05):
    """
    Compute critical infection rate for SIR: beta_crit = mu / lambda_max(DB)
    """
    lambda_max = eigs(DB, k=1, which='LR', return_eigenvectors=False)[0].real
    beta_crit = mu / lambda_max if lambda_max > 0 else np.inf
    return lambda_max, beta_crit

# Edge size distribution helper
def edge_size_distribution(H):
    sizes = [len(H.edges.members(e)) for e in H.edges]
    counts = Counter(sizes)
    max_size = max(counts)
    distribution = [counts.get(i, 0) for i in range(max_size + 1)]
    return distribution


# # Generate a random simplicial hypergraph
# def generate_hypergraph(edges_of_size, degree_seq, q):
#     """
#     Generate a simplicial Chung–Lu hypergraph with given q.
#     The q parameter controls the simplicial ratio (probability of higher-order closure).
#     """
#     max_dim = len(edges_of_size)  # ensures proper vector length
#     q_vector = [q] * max_dim      # fix: make q a list so q[k] works in model
#     # edges = hm.simplicial_chung_lu(degree_seq, edges_of_size, q_vector, multisets=True)
#     # Ensure q is iterable for all simplex orders
#     if np.isscalar(q):
#         q = np.ones(len(edges_of_size)) * q
#     edges = hm.simplicial_chung_lu(degree_seq, edges_of_size, q, multisets=True)

#     H = xgi.Hypergraph(edges)
#     return H

def generate_simplicial_hypergraph(nodes, edges_of_size, degree_seq):
    #edges = hm.chung_lu(nodes, edges_of_size, degree_seq, multisets=True)
    edges = hm.simplicial_chung_lu(degree_seq, edges_of_size, 0.5, multisets=True)
    H = xgi.Hypergraph(edges)
    return H


# Sweep over q and compute β_crit
def compute_Bcrit_for_q(num_nodes, edges, degree_seq, gamma, num_graphs):
    """
    Compute and plot critical infection rates B_crit as a function of q (simplicial ratio).
    """
    beta_crits = []
    
    for i in range(num_graphs):
            print(f"Simulation {i+1}/{num_graphs}")
            H = generate_simplicial_hypergraph(num_nodes, edges, degree_seq)
            DB, _ = compute_DB_matrix(H)
            _, beta_crit = compute_sir_threshold(DB, mu=gamma)
            beta_crits.append(beta_crit)

    print(beta_crits)

# Main script
if __name__ == "__main__":
    # Example usage:
    # python b_crit.py 1  → loads dataset index 1 ("contact-high-school")

    dataset = datasets[int(sys.argv[1])]
    H = xgi.load_xgi_data(dataset)
    num_nodes = H.num_nodes
    edges = edge_size_distribution(H)
    degree_seq = [H.degree(n) for n in H.nodes]
    gamma = 0.05
    num_graphs = 10

    # Compute & plot B_crit across q
    compute_Bcrit_for_q(num_nodes, edges, degree_seq, gamma, num_graphs)