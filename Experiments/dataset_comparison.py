import sys
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

def generate_hypergraph(nodes, edges_of_size, degree_seq):
    #edges = hm.chung_lu(nodes, edges_of_size, degree_seq, multisets=True)
    edges = hm.simplicial_chung_lu(degree_seq, edges_of_size, 0.5, multisets=True)
    H = xgi.Hypergraph(edges)
    return H

def edge_size_distribution(H):
    sizes = [len(H.edges.members(e)) for e in H.edges]
    counts = Counter(sizes)
    max_size = max(counts)
    distribution = [counts.get(i, 0) for i in range(max_size + 1)]
    return distribution


def run_multiple_SIR_with_errorbands(
    num_nodes, 
    edges, 
    degree_seq,
    gamma,
    colors,
    num_graphs,
    rho=0.1,
    tmin=0,
    tmax=100,
    dt=1,
    ax=None, 
):
    S_all, I_all, R_all = [], [], []
    t_vals = None  # to store time vector from first run

    for i in range(num_graphs):
        print(f"Simulation {i+1}/{num_graphs}")
        H = generate_hypergraph(num_nodes, edges, degree_seq)

        tau = {k: (0.1/k) for k in xgi.unique_edge_sizes(H)}
        t, S, I, R = hc.discrete_SIR(H, tau, gamma=gamma, rho=rho, tmin=tmin, tmax=tmax, dt=dt)

        if t_vals is None:
            t_vals = t  # save the time vector
        
        min_len = min(len(S), len(I), len(R))
        S_all.append(S[:min_len] / num_nodes)
        I_all.append(I[:min_len] / num_nodes)
        R_all.append(R[:min_len] / num_nodes)

        if t_vals is None:
            t_vals = t[:min_len]

    min_len = min(len(arr) for arr in S_all)  # Find the minimum time series length

    # Trim all arrays to min_len
    S_all = np.array([s[:min_len] for s in S_all])
    I_all = np.array([i[:min_len] for i in I_all])
    R_all = np.array([r[:min_len] for r in R_all])
    t_vals = t_vals[:min_len]

    # Compute mean and std
    S_mean, S_std = np.mean(S_all, axis=0), np.std(S_all, axis=0)
    I_mean, I_std = np.mean(I_all, axis=0), np.std(I_all, axis=0)
    R_mean, R_std = np.mean(R_all, axis=0), np.std(R_all, axis=0)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Use ax instead of axes for plotting
    ax.plot(t_vals, S_mean, color = colors[0], label="S (mean)")
    ax.fill_between(t_vals, S_mean - S_std, S_mean + S_std, color = colors[0], alpha=0.3)
    ax.plot(t_vals, I_mean, color = colors[1], label="I (mean)")
    ax.fill_between(t_vals, I_mean - I_std, I_mean + I_std, color = colors[1], alpha=0.3)
    ax.plot(t_vals, R_mean, color = colors[2], label="R (mean)")
    ax.fill_between(t_vals, R_mean - R_std, R_mean + R_std, color = colors[2], alpha=0.3)
    ax.set_ylabel("Fraction of Population")
    ax.set_title(f"SIR on Generated Code Error Bands")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig, ax


def SIR_original_graph(
    dataset,
    gamma,
    colors,
    ax=None
):
    
    H = xgi.load_xgi_data(dataset)
    num_nodes = H.num_nodes
    tau = {i: (0.1/i) for i in xgi.unique_edge_sizes(H)}
    t1, S1, I1, R1 = hc.discrete_SIR(H, tau, gamma, tmin=0, tmax=100, dt=1, rho=0.1)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(t1, S1 / num_nodes, "g--", color = colors[0], label="S (discrete)")
    ax.plot(t1, I1 / num_nodes, "r--", color = colors[1], label="I (discrete)")
    ax.plot(t1, R1 / num_nodes, "b--", color = colors[2], label="R (discrete)")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction of population")
    ax.set_title("SIR on Original Graph")
    ax.grid(True)
    fig.tight_layout()
    return fig, ax
    
# Example usage:
if __name__ == "__main__":
    dataset = datasets[int(sys.argv[1])]
    H = xgi.load_xgi_data(dataset)
    num_nodes = H.num_nodes
    edges = edge_size_distribution(H)
    degree_seq = [H.degree(n) for n in H.nodes]
    gamma = 0.05
    colors = ["#00B388","#DA291C", "#418FDF"]
    num_graphs = 10  # Number of graphs to simulate

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    run_multiple_SIR_with_errorbands(num_nodes, edges, degree_seq, gamma, colors, num_graphs, ax=axes[0])
    SIR_original_graph(dataset, gamma, colors, ax=axes[1])

    fig.suptitle("SIR with Error Bands, Dataset: " + dataset)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()