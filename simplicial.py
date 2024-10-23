import numpy as np
from collections import Counter
from time import time
from utils import *
from chung_lu import *

TRACK_TIME = False

class Simplicial:
    
    def __init__(
        self,
        E,
        pre_process_sample_size=1000,
        top_edge_sample_size=1000,
    ):
        '''
        Computes the various simplicial ratios when given a list of edges (list of lists or list of sets) E.
        
        pre_process_sample_size is the sample size for estimating the probability that a random Chung-Lu edge is simple. 
        
        top_edge_sample_size is the sample size of "top" edges for estimating the expected number of simplicial pairs.
        
        Increasing either sample sizes will improve the accuracy of the results, but will also take longer to run.
        
        IMPORTANT: if your graph has edges of size ~ sqrt(len(V)), the number of times a Chung-Lu edge will need to be resampled will be very large. As a rule of thumb, you should restrict your edge sizes to at most sqrt(len(V))/2.
        '''
        self.E = E
        self.V = set()
        for e in self.E:
            for v in e:
                self.V.add(v)
        self.pre_process_sample_size = pre_process_sample_size
        self.top_edge_sample_size = top_edge_sample_size
        self.counts = None
        self.temporal_counts = None
        self.ratio, self.up_ratio, self.down_ratio, self.matrix, self.temporal_matrix = self._compute_all()

    # We compute the simplicial ratio, matrix, and temporal variants all at once.
    def _compute_all(self):
        t0 = time()
        # up + down = number of pairs. 
        # up are the "bottom-up" pairs, where the small edge came first in E.
        # M is the simplicial matrix and M_ordered pays attention to which edge came first.
        up, down, M, M_ordered = self._simplicial_pairs()
        self.counts = M.copy()
        self.temporal_counts = M_ordered.copy()
        
        t1 = time()
        if TRACK_TIME:
            print('time to count simplicial pairs = {0}'.format(t1-t0))
    
        # We compute M_cl, the expected number of (i,j)-pairs for each i,j. 
        M_cl = expected_pairs(V=self.V, E=self.E, num_pre_process_samples=self.pre_process_sample_size, num_edge_samples=self.top_edge_sample_size)
    
        # Then we use M_cl to compute the other expectations.
        pairs_cl = np.sum(M_cl)
        ratio = (up+down)/pairs_cl
        # In expectation, half of pairs_cl are "bottom-up" pairs, and half are "top-down" pairs.
        up_ratio = 2*up/pairs_cl
        down_ratio = 2*down/pairs_cl
        M_ratio = M.copy()
        M_ordered_ratio = M_ordered.copy()
        sizes = {len(e) for e in self.E}
        for k in range(len(M)):
            for l in range(len(M[0])):
                if M_cl[k][l] != 0:
                    M_ratio[k][l] = M[k][l]/M_cl[k][l]
                    M_ordered_ratio[k][l] = 2*M_ordered[k][l]/M_cl[k][l]
                    M_ordered_ratio[l][k] = 2*M_ordered[l][k]/M_cl[k][l]

        return ratio, up_ratio, down_ratio, M_ratio, M_ordered_ratio

    # Counts the number of simplicial pairs for each i,j.
    def _simplicial_pairs(self):

        # Initializing deg.
        E_with = edge_sets(self.V, self.E)
        deg = {v: len(E_with[v]) for v in self.V}

        # Initializing the counts.
        up = 0
        down = 0
        max_size = max(len(e) for e in self.E)
        M = np.zeros((max_size+1, max_size+1))
        M_ordered = np.zeros((max_size+1, max_size+1))

        # We loop through e in E and find all pairs with e as the smaller edge.
        # This way we find all pairs and avoid double counting.
        for i, e in enumerate(self.E):
            if len(e) < max_size:
                # To find edges containing e, we find the vertex of min degree in e and consider all its edges.
                v = min_deg_in_edge(e, deg)
                # Any edge containing e must contain v. 
                relevant_indices = E_with[v]
                up, down, M, M_ordered = self._update_pairs(i, relevant_indices, up, down, M, M_ordered)
                
        return up, down, M, M_ordered    

    # Adds all pairs with e as the smaller edge
    def _update_pairs(self, e_index, relevant_indices, up, down, M, M_ordered):
        for i in relevant_indices:
            if set(self.E[e_index]) < set(self.E[i]):
                M[len(self.E[e_index])][len(self.E[i])] += 1
                # For temporal counts, we care about which edge is first in E.
                if e_index < i:
                    up += 1
                    M_ordered[len(self.E[e_index])][len(self.E[i])] += 1
                else:
                    down += 1
                    M_ordered[len(self.E[i])][len(self.E[e_index])] += 1   
                    
        return up, down, M, M_ordered

    # Returns all index pairs corresponding to simplicial pairs. 
    # The logic for this function is identical to _simplicial_pairs
    def list_of_simplicial_pairs(self):
        '''
        Returns a list of all pairs (i, j) where i and j are the indices of a simplicial pair in E
        '''
        E_with = edge_sets(self.V, self.E)
        deg = {v : len(E_with[v]) for v in self.V}
        max_size = max(len(e) for e in self.E)
        pairs = []
        for i, e in enumerate(self.E):
            if len(e) < max_size:
                v = min_deg_in_edge(e, deg)
                relevant_indices = E_with[v]
                pairs = self._update_list_of_pairs(i, relevant_indices, pairs)
        return pairs

    def _update_list_of_pairs(self, e_index, relevant_indices, pairs):
        for i in relevant_indices:
            if set(self.E[e_index]) < set(self.E[i]):
                pairs.append([e_index, i])
        return pairs
