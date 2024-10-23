import numpy as np
import math
from itertools import combinations
from time import time
from utils import *

TRACK_TIME = False


# Main function.
# Computes the expected number of (i,j) pairs for each i,j.
def expected_pairs(V, E, num_pre_process_samples=1000, num_edge_samples=1000):
    
    t0 = time()

    # Initializing deg.
    es = edge_sets(V, E)
    deg = {v: len(es[v]) for v in V}

    # Initializing m.
    E_sorted = sort_edges(E)
    m = {k: len(E_sorted[k]) for k in E_sorted}

    # Initializing other useful things.
    sizes = list(m.keys())
    k_min = min(sizes)
    k_max = max(sizes)
    volume = sum(deg.values())

    # p_multiset is needed to compute expected number of pairs.
    p_multiset = pre_process(sizes, deg, num_pre_process_samples)

    # M is what gets returned.
    M = np.zeros((k_max+1, k_max+1))

    # sampled_edges is a dict pointing l to a bunch of l-edges sampled via Chung-Lu.
    # We sample "top" edges and compute the expected number of "bottom" edges, hence the l != k_min part.
    sampled_edges = {l: get_chung_lu_edges(num_edge_samples, l, deg) for l in sizes if l != k_min}
    
    t1 = time()
    if TRACK_TIME:
        print('time to sample chung-lu edges = {0}'.format(t1-t0))

    # For each (k,l), we compute the expected number of (k,l) pairs based on sampled_edges[l] and a combinatorial approximation argument.
    for k in sizes:
        for l in sizes:
            if k<l:
                # p_pair will be the average probability of seeing a k-edge inside an element on sampled_edges[l]
                p_pair = 0
                for e in sampled_edges[l]:
                    p_pair += probability_of_pair(e, k, deg, volume, p_multiset)
                p_pair = p_pair/len(sampled_edges[l])
                # num_expected is the probability of a pair times the number of possible pairs
                num_expected = m[k]*m[l]*p_pair
                M[k][l] = num_expected
                
    t2 = time()
    if TRACK_TIME:
        print('time to compute expected matrix = {0}'.format(t2-t1))
        
    return M


# The next three functions generate Chung-Lu edges.
def get_samples(num, deg):
    p = normalize(list(deg.values()))
    return np.random.choice(a=list(deg.keys()), size=num, p=p)

def get_next_edge(num, deg, size, samples, pointer, max_pointer):
    e = samples[pointer: pointer+size]
    # If size is ~ sqrt(len(V)), the expected number of repeats in your edge is > 0.
    # So sampling huge edges is a problem. 
    while len(e) != len(set(e)):
        pointer += size
        if pointer >= max_pointer:
            samples = get_samples(num*size*size, deg)
            pointer = 0
        e = samples[pointer: pointer+size]
    pointer += size
    if pointer >= max_pointer:
        samples = get_samples(num*size*size, deg)
        pointer = 0
    return e, samples, pointer

def get_chung_lu_edges(num, size, deg):
    samples = get_samples(num*size*size, deg)
    edges = [[] for i in range(num)]
    pointer = 0
    max_pointer = num*size
    for i in range(num):
        e, samples, pointer = get_next_edge(num, deg, size, samples, pointer, max_pointer)
        edges[i].extend(e)
    return edges


# Estimates the probability of generating a multiset edge in the Chung-Lu model.
def probability_of_multiset(deg, num, size):
    top = 0
    samples = get_samples(num*size, deg)
    for i in range(num):
        e = samples[i*size: (i+1)*size]
        if len(e) != len(set(e)):
            top += 1
    return top/num

# p_multiset[k] is the probability that a k-edge is a multiset edge.
def pre_process(sizes, deg, num):
    p_multiset = {}
    for k in sizes:
        p = probability_of_multiset(deg, num, k)
        p_multiset[k] = p
    return p_multiset

# Computes the probability that "edge" encapsulates a random k-edge.
def probability_of_pair(e, k, deg, volume, p_multiset):
    e_degs = [deg[v] for v in e]
    k_set_weights = [math.prod(t) for t in combinations(e_degs, k)]
    e_weight = math.factorial(k)*sum(k_set_weights)
    total_weight = (1-p_multiset[k])*(volume**k)
    return e_weight/total_weight


