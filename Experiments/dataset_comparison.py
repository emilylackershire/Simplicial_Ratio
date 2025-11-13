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
from termcolor import colored
from trie import Trie
from utilities import count_subfaces, max_number_of_subfaces, missing_subfaces

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

def new_edit_simpliciality(H, min_size=2, exclude_min_size=True):
    """Computes the simplicial edit distance.

    The number of edges needed to be added
    to a hypergraph to make it a simplicial complex.

    Parameters
    ----------
    H : xgi.Hypergraph
        The hypergraph of interest
    min_size: int, default: 1
        The minimum hyperedge size to include when
        calculating whether a hyperedge is a simplex
        by counting subfaces.

    Returns
    -------
    int
        The edit simpliciality
    """
    edges = H.edges.filterby("size", min_size, "geq").members()
    max_edges = H.edges.maximal().filterby("size", min_size, "geq").members()
    t = Trie()
    t.build_trie(edges)

    maxH = xgi.Hypergraph(
        H.edges.maximal()
        .filterby("size", min_size + exclude_min_size, "geq")
        .members(dtype=dict)
    )
    if not maxH.edges:
        return np.nan

    ms = 0
    for id1, e in maxH.edges.members(dtype=dict).items():
        redundant_missing_faces = set()
        for id2 in maxH.edges.neighbors(id1):
            if id2 < id1:
                c = maxH._edge[id2].intersection(e)
                if len(c) >= min_size:
                    redundant_missing_faces.update(missing_subfaces(t, c, min_size))

                    # we don't have to worry about the intersection being a max face
                    # because a) there are no multiedges and b) these are all maximal
                    # faces so no inclusions.
                    if not t.search(c):
                        redundant_missing_faces.add(frozenset(c))

        nm = max_number_of_subfaces(min_size, len(e))
        nf = count_subfaces(t, e, min_size)
        rmf = len(redundant_missing_faces)
        ms += nm - nf - rmf

    try:
        s = len(edges)
        return (s - len(max_edges)) / (ms + s - len(max_edges))
    except ZeroDivisionError:
        return np.nan


def generate_simplicial_hypergraph(nodes, edges_of_size, degree_seq):
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
    dataset_index,
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

    #
    #total statistics for the table
    avg_ES = 0
    avg_nodes = 0
    avg_edges = 0
    avg_max_he = 0
    avg_clust_coef = 0
    avg_degree = 0
    avg_degree_assot = 0
    #

    for i in range(num_graphs):
        print(f"Simulation {i+1}/{num_graphs}")
        H = generate_simplicial_hypergraph(num_nodes, edges, degree_seq)

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

        avg_ES += new_edit_simpliciality(H)
        avg_nodes += H.num_nodes
        avg_edges += H.num_edges
        avg_max_he += max(len(H.edges.members(e)) for e in H.edges)
        avg_clust_coef += sum(list(xgi.clustering_coefficient(H).values())) / H.num_nodes
        #to get avg degree, we make a deg_list that takes the count of degrees per each node
        deg_list = list(xgi.degree_counts(H))
        degrees = []
        #iterates through deg_list with enumerate, which tells us the degree val and number of nodes with that degree
        for degree_val, count in enumerate(deg_list):
            #multiplies count of degrees with current val to get sum
            degree = count * degree_val
            #appends to degrees array
            degrees.append(degree)
        #sums list and divides by number of nodes to get average
        avg_degree += sum(degrees) / H.num_nodes
        avg_degree_assot += xgi.degree_assortativity(H)

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
    ax.set_title(f"SIR on simplicial chung lu Generated Code Error Bands")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    total_ES = round((avg_ES / num_graphs), 5)
    total_nodes =  round((avg_nodes / num_graphs), 5)
    total_edges =  round((avg_edges / num_graphs), 5)
    total_max_he =  round((avg_max_he / num_graphs), 5)
    total_clust_coef =  round((avg_clust_coef / num_graphs), 5)
    total_degree =  round((avg_degree / num_graphs), 5)
    total_degree_assot =  round((avg_degree_assot / num_graphs), 5)

    print(colored("ES: " + str(total_ES) + "\n" + 
            "Average nodes: " + str(total_nodes) + "\n" +
            "Average edges: " + str(total_edges) + "\n" +
            "Average max hyperedge size: " + str(total_max_he) + "\n" +
            "Average clustering coefficient: " + str(total_clust_coef) + "\n" +
            "Average degree: " + str(total_degree) + "\n" +
            "Average degree assortativity: " + str(total_degree_assot) + "\n", "green"))
    
    print(colored( "chung lu " + str(dataset_index) +  "& " +
                    str(total_ES) + "&" +
                    str(total_nodes) + "&" +
                    str(total_edges) + "&" +
                    str(total_max_he) + "&" +
                    str(total_clust_coef) + "&" +
                    str(total_degree) + "&" +
                    str(total_degree_assot) + "\\\\", "red"))
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
    


if __name__ == "__main__":
    dataset = datasets[int(sys.argv[1])]
    H = xgi.load_xgi_data(dataset)
    #update data so we take the modified version, with no duplicate edges, no directed edges, and only looking at the largest connected component
    H_cleaned = H.cleanup(multiedges=False, singletons=False, isolates=False, relabel=True, in_place=False)
    cleaned_ES = round((new_edit_simpliciality(H_cleaned)), 5)
    num_nodes = H_cleaned.num_nodes
    cleaned_edges = edge_size_distribution(H_cleaned)
    total_edges = H_cleaned.num_edges
    degree_seq = [H_cleaned.degree(n) for n in H_cleaned.nodes]
    gamma = 0.05
    colors = ["#00B388","#DA291C", "#418FDF"]
    num_graphs = 10  # Number of graphs to simulate

    max_he = round((max(len(H_cleaned.edges.members(e)) for e in H_cleaned.edges)), 5)
    clust_coef = round((sum(list(xgi.clustering_coefficient(H_cleaned).values())) / H_cleaned.num_nodes), 5)
    #to get avg degree, we make a deg_list that takes the count of degrees per each node
    deg_list = list(xgi.degree_counts(H_cleaned))
    degrees = []
    #iterates through deg_list with enumerate, which tells us the degree val and number of nodes with that degree
    for degree_val, count in enumerate(deg_list):
        #multiplies count of degrees with current val to get sum
        degree = count * degree_val
        #appends to degrees array
        degrees.append(degree)
    #sums list and divides by number of nodes to get average
    avg_degree = round((sum(degrees) / H_cleaned.num_nodes), 5)
    degree_assot = round((xgi.degree_assortativity(H_cleaned)),5)

    print(colored("ES: " + str(cleaned_ES) + "\n" + 
            "Average nodes: " + str(num_nodes) + "\n" +
            "Average edges: " + str(total_edges) + "\n" +
            "Average max hyperedge size: " + str(max_he) + "\n" +
            "Average clustering coefficient: " + str(clust_coef) + "\n" +
            "Average degree: " + str(avg_degree) + "\n" +
            "Average degree assortativity: " + str(degree_assot) + "\n", "green"))
    
    print(colored( "real " + str(dataset) +  "& " +
                    str(cleaned_ES) + "&" +
                    str(num_nodes) + "&" +
                    str(total_edges) + "&" +
                    str(max_he) + "&" +
                    str(clust_coef) + "&" +
                    str(avg_degree) + "&" +
                    str(degree_assot) + "\\\\", "blue"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    run_multiple_SIR_with_errorbands(int(sys.argv[1]), num_nodes, cleaned_edges, degree_seq, gamma, colors, num_graphs, ax=axes[0])
    SIR_original_graph(dataset, gamma, colors, ax=axes[1])

    fig.suptitle("SIR with Error Bands, Dataset: " + dataset)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()