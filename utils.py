import numpy as np

# E_with[v] is a list of edges containing v
def edge_sets(V, E):
    E_with = {v: [] for v in V}
    for i, e in enumerate(E):
        for v in e:
            E_with[v].append(i)
    return E_with

def min_deg_in_edge(e, deg):
    e_deg = {v: deg[v] for v in e}
    v = min(e_deg, key = e_deg.get)
    return v

def normalize(X):
    norm = sum(X)
    return [x/norm for x in X]

# E_sorted[k] is a list of k-edges
def sort_edges(E):
    sizes = {len(e) for e in E}
    E_sorted = {k: [] for k in sizes}
    for i, e in enumerate(E):
        E_sorted[len(e)].append(i)
    return E_sorted

# This is the fastest dataset for retriving connected components.
class UnionFind:
    def __init__(
        self,
        V,
    ):
        self.V = V
        self.parents = {v: v for v in self.V}
        self.sizes = {v: 1 for v in self.V}

    # Returns the root of a vertex.
    def root(self, v):
        if self.parents[v] == v:
            return v
        else:
            return self.root(self.parents[v])

    # Makes root[v] the same for all v in S.
    def merge(self, S):  
        S = {self.root(v) for v in S}
        if len(S) > 1:
            v_max = list(S)[0]
            for v in S:
                if self.sizes[v] > self.sizes[v_max]:
                    v_max = v
            for v in S:
                self.parents[v] = v_max
            self.sizes[v_max] = sum([self.sizes[v] for v in S])
        return 

# Returns a partition of V into connected components
def components(V, E):
    uf = UnionFind(V)
    for e in E:
        uf.merge(e)
    roots = {uf.root(v) for v in V}
    components = {v : {v} for v in roots}
    for v in V:
        components[uf.root(v)].add(v)
    return list(components.values())

# Returns the vertices and edges of the largest component
def giant_component(V, E):
    comps = components(V, E)
    C_max = comps[0]
    for C in comps:
        if len(C) > len(C_max):
            C_max = C
    E = [e for e in E if list(e)[0] in C_max]
    V = C_max
    V = list(V)
    return V, E

# This returns a nicer looking simplicial matrix
def make_matrix_pretty(M):
    M_pretty = M.astype(str)
    M_pretty = np.delete(M_pretty, 0, axis=0)
    M_pretty = np.delete(M_pretty, 0, axis=0)
    M_pretty = np.delete(M_pretty, 0, axis=1)
    M_pretty = np.delete(M_pretty, 0, axis=1)
    for i in range(len(M_pretty)):
        for j in range(len(M_pretty[i])):
            if float(M_pretty[i][j]) > 1000:
                M_pretty[i][j] = ">1k"
            elif float(M_pretty[i][j]) >= 100:
                M_pretty[i][j] = str(round(float(M_pretty[i][j])))
            else:
                M_pretty[i][j] = str(round(float(M_pretty[i][j]), 1))
    return M_pretty