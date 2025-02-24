import numpy as np
import random
from math import comb
from itertools import combinations as combs
import Hypergraph_Functions as hf
import time
from collections import Counter



##########Notes##########
# - Before each function is a short description, followed by an explanation of each input
# - Vertices should always be a set or a list of hashable elements.
# - Edges should always be a list of vertices.
# - The functions only return hyperedges, not vertices
#########################



##########Models##########
#The following models are currently available:
#erdos_renyi
#chung_lu
#configuration_model
##########################


#Generates the edges of an Erdos-Renyi hypergraph

#V: Either a set of vertices or the number of vertices
#m: Either a list of edges or a list [m_1,m_2,...,m_k] where m_i is the number of i-edges

def erdos_renyi(V,m):
    #If V is a number, we convert to a list
    if type(V) is int:
        V = list(range(V))
    #We convert V to a list so that we can use np.random.choice
    else:
        V = list(V)

    #If m is a list of edges, we get the number of edges of each size
    if (type(m[0]) is set) or (type(m[0]) is list):
        #sort edges by size
        edge_dict = hf.sort_edges(m)

        #convert m to a list of num edges per size
        sizes = edge_dict.keys()       
        m = [0]*max(sizes)
        for k in sizes:
            m[k-1] = len(edge_dict[k])

    #Initialize edges
    E_by_size = {}
    E = []

    #Iterate through edge sizes and build edges
    for k in range(1,len(m)+1):
        if m[k-1] > 0:
            #If we are to generate more than 50% of edges of a given size, we instead build non-edges
            remove = False
            if m[k-1] > comb(len(V),k)/2:
                remove = True
                m[k-1] = comb(len(V),k) - m[k-1]

            #Initialize edges of size k
            E_by_size[k] = set()
            #loop until we have enough edges built
            while(len(E_by_size[k]) < m[k-1]):
                #By storing a set of frozensets, we throw away duplicate edges
                new_edge = frozenset(np.random.choice(V,k,replace = False))
                E_by_size[k].add(new_edge)
            
            #Here we flip edges to non-edges if needed
            if remove:
                E_by_size[k] = {frozenset(e) for e in combs(V,k)} - E_by_size[k]

            #Finally, we convert back to a list of sets and add to E
            E_by_size[k] = list(map(set, list(E_by_size[k])))
            E += E_by_size[k]

    return E


#Generates the edges of a Chung-Lu hypergraph

#V: Either a set of vertices or a number of vertices
#m: Either a list of edges or a list of num edges per size
#degrees: Either a list or a dictionary of degrees.

#Note: Acceptable inputs are
# - vertices and edges (V,E)
# - vertices, num edges per size, and degree dict (V,m,degree dict)
# - int, num edges per size, and degree list (n,m,degree list)

def chung_lu(vertices,m,degrees = None, multisets = True):
    
    #If V is a number, we convert to a list
    if type(vertices) is int:
        V = list(range(vertices))
    else:
        V = list(vertices.copy())

    #If m is a list of edges, we get the degrees and the number of edges of each size
    if (type(m[0]) is set) or (type(m[0]) is list):
        degrees = hf.degrees(V,m.copy())
        
        #sort edges by size
        edge_dict = hf.sort_edges(m.copy())

        #convert m to a list of num edges per size
        sizes = edge_dict.keys()
        m = [0]*max(sizes)
        for k in sizes:
            m[k-1] = len(edge_dict[k])

    #If degrees is a list, we convert to a dictionary
    #Note: If V was given as a set, and degrees as a list of degrees, then the degrees might get shuffled
    if type(degrees) is list:
        degrees = dict(zip(V,degrees))
    L = hf.volume(V,degrees)
    
    #choices is a dictionary with degrees[v] keys pointing to v
    #I've tested, and this is much faster than making a list 
    choices = dict.fromkeys(set(range(L)))
    counter = 0
    current_vertex = 0
    #We need L keys in total
    while (counter<L):
        for i in range(degrees[V[current_vertex]]):
            choices[counter] = V[current_vertex]
            counter += 1
        current_vertex += 1

    #E is the set of edges to be returned
    E = []
    if multisets:
        for k in range(len(m)):
            #Adding all edges of size k+1        
            for i in range(m[k]):
                e = []
                for j in range(k+1):
                    e.append(choices[random.randint(0,L-1)])
                E.append(e)
    else:
        for k in range(len(m)):
            #Adding all edges of size k+1        
            for i in range(m[k]):
                e = []
                while len(e)<k+1:
                    v = (choices[random.randint(0,L-1)])
                    if v not in e:
                        e.append(v)
                E.append(e)
    return E


def connected_chung_lu(vertices,m,degrees = None, multisets = True):

    ##
    #CLEANING THE INPUT
    ##
    
    #If V is a number, we convert to a list
    if type(vertices) is int:
        V = list(range(vertices))
    else:
        V = list(vertices.copy())

    #If m is a list of edges, we get the degrees and the number of edges of each size
    if (type(m[0]) is set) or (type(m[0]) is list):
        degrees = hf.degrees(V,m.copy())
        
        #sort edges by size
        edge_dict = hf.sort_edges(m.copy())

        #convert m to a list of num edges per size
        sizes = edge_dict.keys()
        m = [0]*max(sizes)
        for k in sizes:
            m[k-1] = len(edge_dict[k])

    #If degrees is a list, we convert to a dictionary
    #Note: If V was given as a set, and degrees as a list of degrees, then the degrees might get shuffled
    if type(degrees) is list:
        degrees = dict(zip(V,degrees))
    L = hf.volume(V,degrees)

    #We normalize the degree sequence so we can take random samples
    normalized_degrees = {}
    for v in degrees.keys():
        normalized_degrees[v] = degrees[v]/L


    ##
    #BUILDING THE UNION-FIND DATASET
    ##
    
    parents = {v : v for v in V}
    sizes = {v : normalized_degrees[v] for v in V}
    
    #Inernal function: Finds the current root of a vertex
    def root(v):
        if parents[v] == v:
            return v
        else:
            return root(parents[v])

    #Internal function: Merges sets
    def merge(S):  
        #replace elements by their roots
        S = {root(v) for v in S}

        #If all vertices were in the same component, we don't need to merge
        if len(S) > 1:
            #When we merge, we root using the largest tree
            v = next(iter(S))
            for u in S:
                if sizes[u] > sizes[v]:
                    v = u

            #We now point all of S to v
            for u in S:
                parents[u] = v

            #Lastly, we update the size of v
            #Note: We only care about the sizes of roots, so there is no need to update the size of S\{v}
            #Note: We converted S to a set of roots, so this sum is always correct
            sizes[v] = sum([sizes[u] for u in S])
    
    #We build a grab-bag of edge-size choices
    size_choices = []
    for i,m_i in enumerate(m):
        #This is only for connecting the graph, so we don't allow singletons
        if i>0:
            for j in range(m_i):
                size_choices.append(i+1)
    random.shuffle(size_choices)

    E = []
    num_components = len(V)
    while num_components > 1:
        edge_size = size_choices.pop()
        m[edge_size-1] -= 1
        roots = list({root(v) for v in V})
        weights = [sizes[v] for v in roots]
        if num_components < edge_size:
            components_to_merge = list(np.random.choice(roots,num_components,p=weights,replace=False))
            #We pick a vertex from each component based on conditional weights
            chosen_vertex = {}
            for v in components_to_merge:
                #TODO: speed up by building the collection of components beforehand
                vertices_to_choose_from = {u for u in V if root(u)==v}
                component_weight = {u:degrees[u] for u in vertices_to_choose_from}
                total_weight = sum(component_weight.values())
                for u in component_weight.keys():
                    component_weight[u] = component_weight[u]/total_weight
                chosen_vertex[v] = np.random.choice(list(component_weight.keys()),p=list(component_weight.values()))
            e = [chosen_vertex[v] for v in components_to_merge]            
            merge(components_to_merge)
            if multisets:
                e_remaining = list(np.random.choice(list(normalized_degrees.keys()),edge_size-num_components,p=list(normalized_degrees.values()),replace=True))
                e.extend(e_remaining)
            else:
                while True:
                    e_try = list(np.random.choice(list(normalized_degrees.keys()),num_components-1,p=list(normalized_degrees.values()),replace=False))
                    if len(set(e).intersection(set(e_try))) == 0:
                        e.extend(e_try)
                        break
            num_components = 1
        else:            
            components_to_merge = list(np.random.choice(roots,edge_size,p=weights,replace=False))          
            #We pick a vertex from each component based on conditional weights
            chosen_vertex = {}
            for v in components_to_merge:
                #TODO: speed up by building the collection of components beforehand
                vertices_to_choose_from = {u for u in V if root(u)==v}
                component_weight = {u:degrees[u] for u in vertices_to_choose_from}
                total_weight = sum(component_weight.values())
                for u in component_weight.keys():
                    component_weight[u] = component_weight[u]/total_weight
                chosen_vertex[v] = np.random.choice(list(component_weight.keys()),p=list(component_weight.values()))
            e = [chosen_vertex[v] for v in components_to_merge]
            merge(components_to_merge)
            num_components -= (edge_size-1) 
        E.append(e)
        
    E_remaining = chung_lu(V,m,degrees,multisets=multisets)
    E.extend(E_remaining)
    return E


#Generates the edges of a configuration model

#V: Either a set of vertices or a number of vertices
#m: Either a list of edges or a list of num edges per size
#degrees: Either a list or a dictionary of degrees.

#Note: Acceptable inputs are
# - vertices and edges (V,E)
# - vertices, num edges per size, and degree dict (V,m,degree dict)
# - int, num edges per size, and degree list (n,m,degree list)

def configuration_model(V,m,degrees = None):
    #If V is a number, we convert to a list
    if type(V) is int:
        V = list(range(V))
    else:
        V = list(V)

    #If m is a list of edges, we get the degrees and the number of edges of each size
    if (type(m[0]) is set) or (type(m[0]) is list):
        degrees = hf.degrees(V,m)
        
        #sort edges by size
        edge_dict = hf.sort_edges(m)

        #convert m to a list of num edges per size
        sizes = edge_dict.keys()       
        m = [0]*max(sizes)
        for k in sizes:
            m[k-1] = len(edge_dict[k])

    #If degrees is a list, we convert to a dictionary
    #Note: If V was given as a set, and degrees as a list of degrees, then the degrees might get shuffled
    if type(degrees) is list:
        degrees = dict(zip(V,degrees))
    L = hf.volume(V,degrees)
    
    #choices is a list with degrees[v] copies of v for each v in V
    choices = []
    counter = 0
    current_vertex = 0
    #We need L keys in total
    while (counter<L):
        for i in range(degrees[V[current_vertex]]):
            choices.append(V[current_vertex])
            counter += 1
        current_vertex += 1
    
    #E is the set of edges to be returned
    E = []
    #We shuffle choices and pop elements one by one
    random.shuffle(choices)
    for k in range(len(m)):
        #Adding all edges of size k+1
        for i in range(m[k]):
            e = []
            for j in range(k+1):
                e.append(choices.pop())
            E.append(e)

    return E


################
###ABCD Model###
################


def ABCD(n,gamma,delta,zeta,beta,s,tau,xi,rewire=True):

    start_time = time.time()
    #Fixing input parameters
    #I might add an initial check to make sure parameters are fiesable
    if type(n) is not int:
        n = len(n)
    V = set(range(n))
    if type(zeta) is int:
        max_degree = zeta
    else:
        max_degree = int(np.floor(n**zeta))
    if type(tau) is int:
        max_size = tau
    else:    
        max_size = int(np.floor(n**tau))

    #Phase 1: creating a degree distribution
    #The degree distribution is a truncated power-law with parameter gamma and truncation delta/max_degree
    available_degrees = list(range(delta,max_degree+1))
    degree_distribution = []
    normalization = sum(d**(-gamma) for d in available_degrees)
    for d in available_degrees:
        degree_distribution.append((d**(-gamma))/normalization)
    degree_sequence = sorted(np.random.choice(available_degrees,n,p=degree_distribution))
    #If the degree is odd, we add 1 to the max degree
    if sum(degree_sequence)%2 == 1:
        degree_sequence[-1] = degree_sequence[-1]+1

    #Phase 2: creating a community size distribution
    available_sizes = list(range(s,max_size+1))
    size_distribution = []
    normalization = sum(c**(-beta) for c in available_sizes)
    for c in available_sizes:
        size_distribution.append((c**(-beta))/normalization)
    #We generate community sizes until the total is >= n
    #Since np.random.choice is slow if called many times, we get a big list and then truncate
    #The number of communities cannot exceed n/s, so we generate that many
    temp_sequence = list(np.random.choice(available_sizes,int(np.ceil(n/s)),p=size_distribution))
    size_sequence = []
    sum_of_sizes = 0
    while(sum_of_sizes < n):
        new_size = temp_sequence.pop()
        size_sequence.append(new_size)
        sum_of_sizes += new_size
    overflow = sum_of_sizes - n
    if overflow > 0:
        #If overflow > 0 and the last size added is at least s+overflow, we subtract overflow from this size
        if new_size >= (s+overflow):
            size_sequence[-1] -= overflow
        #Otherwise, we delete this community and add 1 to communities at random until the sum is n
        else:
            size_sequence.pop()
            random.shuffle(size_sequence)
            for i in range(new_size - overflow):
                size_sequence[i] += 1
    size_sequence = sorted(size_sequence)
    num_communities = len(size_sequence)
    #Lastly, we partition the vertices into communities
    #At first, communities will be a dict of label -> vertices
    #Later, this will be changed to label -> vertices and edges
    communities = {C : set() for C in range(num_communities)}
    current_vertex = 0
    #We can now partition the vertices into communities
    for C in communities.keys():
        for i in range(size_sequence[C]):
            communities[C].add(current_vertex)
            current_vertex += 1
    
    #Phase 3: assigning degrees to nodes
    #Starting from the largest degree, we lock some communities if they are too small,
    #then choose a random unlocked vertex and give it the current degree
    degrees = {}
    #Read the ABCD paper to understand phi better
    #It's an error term that controls background edges that happen to land in a community
    phi = 1 - np.sum(np.array([c**2 for c in size_sequence]))/n**2
    #When we unlock a community, we add all of its vertices to available_vertices
    #When we assign a degree to a vertex, we remove it from available_vertices
    available_vertices = []
    #lock determines which communities are off limits when assigning a degree
    lock = num_communities
    #We iterate through degrees, from largest to smallest
    for d in reversed(degree_sequence):
        #A degree can only be assigned to a vertex in C if |C| >= min_size
        #Read the ABCD paper to understand this lowerbound
        min_size = d*(1-xi*phi)+1
        locked_sizes = size_sequence[0:lock].copy()
        #We iterate through locked_sizes, find all sizes that are now available, and update new_lock accordingly
        new_lock = lock
        for c in reversed(locked_sizes):
            if c<min_size:
                break
            new_lock -= 1
        #range(new_lock,lock) are the newly available communities
        for C in range(new_lock,lock):
            available_vertices.extend(communities[C])
        #We assign d to a random vertex in available_vertices
        #We use pop to also remove the vertex from availbale_vertices
        degrees[available_vertices.pop(random.randrange(len(available_vertices)))] = d
        lock = new_lock
    
    #Phases 4&5: creating edges and rewiring
    #We define a rounding function which preserves expectation
    def rand_round(x):
        p = x-np.floor(x)
        if random.uniform(0,1) <= p:
            return int(np.floor(x)+1)
        else:
            return int(np.floor(x))
    #We now split degrees into community_degrees and background_degrees
    community_degrees = dict.fromkeys(V)
    for v in V:
        community_degrees[v] = rand_round((1-xi)*degrees[v])
    #We check each community for an odd volume
    #If yes, we add 1 to a max degree node
    for C in communities.keys():
        C_degrees = {v : community_degrees[v] for v in communities[C]}
        vol_C = sum(C_degrees.values())
        if vol_C%2 == 1:
            v_max = max(C_degrees, key=C_degrees.get)
            community_degrees[v_max] += 1
            #If xi is small enough, it's possible that the community degree becomes too large
            #So we check for this and subtract 1 instead if that's the case
            if community_degrees[v_max] > degrees[v_max]:
                community_degrees[v_max] -= 2
    #Now we build background_degrees, which is just the remainer for each vertex
    background_degrees = {v : degrees[v]-community_degrees[v] for v in V}
    #Before building graphs, we will define a function that quickly finds multi-edges
    #In most cases, we will have a small set containing all edges that could potentially be multi-edges
    #From here until the end of the code, I flip between sets, frozensets, and lists seemingly at random
    #This is because I'm bad at coding
    def find_dupes(X,look_list = 'all'):
        if look_list == 'all':
            X_frozen = [frozenset(sorted(e)) for e in X if len(set(e))==2]
        else:
            X_frozen = [frozenset(sorted(e)) for e in X if (set(e) in look_list and len(set(e)) == 2)]
        S = set()
        dupes = []
        for e in X_frozen:
            if e in S:
                dupes.append(set(e))
            else:
                S.add(e)
        return dupes
            
    #We can now build all of the community graphs, and the background graph, as configuration models
    #If rewire=True, we rewire communities as we build. Otherwise, we leave loops and multi-edges in
    E = []
    for C in communities.keys():
        community_dict = {v : community_degrees[v] for v in communities[C]}
        community_edges = configuration_model(community_dict.keys(),[0,int(sum(community_dict.values())/2)],community_dict)
        #Life is easier when edges are sorted
        community_edges = [sorted(e) for e in community_edges]
        if rewire:
            #We build recycle_list containing all loops and multi-edges
            #recycle_list -> edges that will be rewired
            #dupes -> set(e) for all e contributing to a multi-edge
            #dupes_found -> updates as we iterate through community_edges
            #The first time we find a dupe, we keep it and add the edge to dupes_found
            #Then, every other time we see that edge, we add it to recycle_list
            recycle_list = []
            dupes = find_dupes(community_edges)
            dupes_found = []
            for e in community_edges:
                if len(set(e)) == 1:           
                    recycle_list.append(e)
                elif set(e) in dupes:
                    if set(e) in dupes_found:
                        recycle_list.append(e)
                    else:
                        dupes_found.append(set(e))
            #Now we rewrire all of recycle_list and repeat until recycle_list is either empty or doesn't decrease in size
            while len(recycle_list) > 0:
                #As we rewire, we keep track of the new edges built
                #These new edges are the only ones that can lead to further issues
                look_list = []
                e_index = -1
                for e in recycle_list:
                    e_index += 1
                    if e != 'skip':                        
                        f = random.choice([x for x in community_edges if x != e])                        
                        #If we happen to pick another edge in recycle_list, we mark it and skip it in the loop
                        #We only need to label `skip' if f is later in the list than e
                        if f in recycle_list[e_index:]:
                            #This is convoluted nonsense and should be ignored
                            recycle_list[recycle_list[e_index:].index(f)+e_index] = 'skip'                        
                        community_edges.remove(e)
                        community_edges.remove(f)
                        community_edges.append(sorted([e[0],f[0]]))
                        community_edges.append(sorted([e[1],f[1]]))                       
                        look_list.append({e[0],f[0]})
                        look_list.append({e[1],f[1]})
                #After rewiring, we make new_recycle_list in the same way
                new_recycle_list = []
                #We now specify look_list to speed up the dupe finding process
                dupes = find_dupes(community_edges,look_list)
                dupes_found = []
                for e in community_edges:
                    if set(e) in look_list:
                        if len(set(e)) == 1:
                            new_recycle_list.append(e)
                        elif set(e) in dupes:
                            if set(e) in dupes_found:
                                new_recycle_list.append(e)
                            else:
                                dupes_found.append(set(e))
                    #If the number of bad edges does not go down, we give up and move the issue to the background graph
                if len(new_recycle_list) >= len(recycle_list):
                    for e in new_recycle_list:
                        community_edges.remove(e)
                        v1 = e.pop()
                        v2 = e.pop()
                        community_degrees[v1] -= 1
                        community_degrees[v2] -= 1
                        background_degrees[v1] += 1
                        background_degrees[v2] += 1
                    break
                recycle_list = new_recycle_list
        E.extend(community_edges)
        communities[C] = {'vertices':communities[C],'edges':community_edges}
    #Finally, we handle the background graph
    background_edges = configuration_model(V,[0,round(sum(background_degrees.values())/2)],background_degrees)
    #Again, this is a temporary fix to an issue with configuration_model
    background_edges = [sorted(e) for e in background_edges if len(e)>0]
    if rewire:
        #We do the same process as we did for each community, except that we don't give up when rewiring
        recycle_list = []
        #We have to deal with a new type of bad edge: e that is already in a community graph
        community_collisions = []
        for e in background_edges:
            if len(set(e))==2:
                #Because we sorted edges, we don't have to worry about list ordering when checking if e in E
                if e in E:
                    community_collisions.append(set(e))
        dupes = find_dupes(background_edges)
        dupes_found = []
        #Now we build recycle_list
        for e in background_edges:
            if len(set(e)) == 1:
                recycle_list.append(e)
            elif set(e) in community_collisions:
                recycle_list.append(e)
            elif set(e) in dupes:
                if set(e) in dupes_found:
                    recycle_list.append(e)
                else:
                    dupes_found.append(set(e))
        #Now we rewrire all of recycle_list and repeat until recycle_list is empty
        while len(recycle_list) > 0:
            look_list = []
            e_index = -1
            for e in recycle_list:
                e_index += 1
                if e != 'skip':
                    f = random.choice([x for x in background_edges if x != e])
                    if f in recycle_list[e_index:]:
                        #This is convoluted nonsense and should be ignored
                        recycle_list[recycle_list[e_index:].index(f)+e_index] = 'skip'
                    background_edges.remove(e)
                    background_edges.remove(f)
                    background_edges.append(sorted([e[0],f[0]]))
                    background_edges.append(sorted([e[1],f[1]]))                       
                    look_list.append({e[0],f[0]})
                    look_list.append({e[1],f[1]})
            #After rewiring, we make new_recycle_list in the same way
            recycle_list = []
            community_collisions = []
            for e in background_edges:
                if set(e) in look_list:
                    if len(set(e))==2:
                        #Because we sorted edges, we don't have to worry about list ordering when checking if e in E
                        if e in E:
                            community_collisions.append(set(e))
            dupes = find_dupes(background_edges,look_list)
            dupes_found = []
            #Now we build recycle_list
            for e in background_edges:
                if set(e) in look_list:
                    if len(set(e)) == 1:
                        recycle_list.append(e)
                    elif set(e) in community_collisions:
                        recycle_list.append(e)
                    elif set(e) in dupes:
                        if set(e) in dupes_found:
                            recycle_list.append(e)
                        else:
                            dupes_found.append(set(e))
    E.extend(background_edges)
    communities['Background Graph'] = {'vertices':V,'edges':background_edges}
    
    end_time = time.time()
    print('run time = {0} seconds'.format(np.round(end_time-start_time)))

    return V,E,communities


def make_edge(V,k,p,multisets=True):
    return list(np.random.choice(V,size=k,replace=multisets,p=p))


#d: degree sequence
#m: dict of edge sizes
#q: dict of forced-pair probabilities
#w: 2d array of weights
def bottom_up_simplicial_chung_lu(d,m,q,w,multisets=True):
    #Initialize
    V = list(d.keys())
    L = sum(d.values())
    p = [d[v]/L for v in V]
    E = {k:[] for k in m.keys()}

    #Build smallest edges normally
    k_min = list(m.keys())[0]
    for i in range(m[k_min]):
        e = make_edge(V,k_min,p,multisets=multisets)
        E[k_min].append(e)
    #E_list is what gets returned at the end
    E_list = E[k_min].copy()

    #Build bigger edges
    for k in list(m.keys())[1:]:
        #q[k] is the probability of forcing a simplicial edge
        #m[k] is the number of edges of size k
        #so num_simplicial can be generated in advance
        num_simplicial = np.random.binomial(m[k],q[k])

        #Non-simplicial edges are built normally
        for i in range(m[k]-num_simplicial):
            e = make_edge(V,k,p,multisets=multisets)
            E[k].append(e)

        #Simplicial edges are built in a particular way
        for i in range(num_simplicial):
            #The first part of the edge is a previously built, smaller edge
            sub_sizes = [j for j in m.keys() if j<k]
            weights = [w[j][k] for j in sub_sizes]
            j = np.random.choice(sub_sizes,p=weights)
            e = E[j][np.random.randint(m[j])].copy()
            #The rest of the edge is built in the usual way
            e_remainder = make_edge(V,k-j,p,multisets=multisets)
            e.extend(e_remainder)
            #If multisets are not allowed, we re-sample until e_sub U e_remainder is a simple edge
            if not multisets:
                while(len(e) > len(set(e))):
                    e = e[:j]
                    e_remainder = make_edge(V,k-j,p,multisets=False)
                    e.extend(e_remainder)
            E[k].append(e)

        #Add E[k] to E_list
        E_list.extend(E[k].copy())

    return E_list


#d: degree sequence
#m: dict of edge sizes
#q: dict of forced-pair probabilities
#w: 2d array of weights
def top_down_simplicial_chung_lu(d,m,q,w,multisets=True):
    #Initialize
    V = list(d.keys())
    L = sum(d.values())
    p = [d[v]/L for v in V]
    E = {k:[] for k in m.keys()}

    #Build largest edges normally
    k_max = list(m.keys())[-1]
    for i in range(m[k_max]):
        e = make_edge(V,k_max,p,multisets=multisets)
        E[k_max].append(e)
    #E_list is what gets returned at the end
    E_list = E[k_max].copy()

    #Build smaller edges
    for k in list(reversed(m.keys()))[1:]:
        #q[k] is the probability of forcing a simplicial edge
        #m[k] is the number of edges of size k
        #so num_simplicial can be generated in advance
        num_simplicial = np.random.binomial(m[k],q[k])

        #Non-simplicial edges are built normally
        for i in range(m[k]-num_simplicial):
            e = make_edge(V,k,p,multisets=multisets)
            E[k].append(e)

        #Simplicial edges are built in a particular way
        for i in range(num_simplicial):
            #We choose a bigger edge to sample from
            sup_sizes = [l for l in m.keys() if l>k]
            weights = [w[k][l] for l in sup_sizes]
            l = np.random.choice(sup_sizes,p=weights)
            e_sup = E[l][np.random.randint(m[l])].copy()
            #Our edge will be a uniform subset of e_sup
            #Note: if multisets are not allowed, there is no choice e will be a multiset
            #So no need to correct for this here
            e = random.sample(e_sup,k)
            E[k].append(e)

        #Add E[k] to E_list
        E_list.extend(E[k].copy())

    return E_list



def fix_inputs_bu(d,m,q,w=None):
    #If d is a list, we label vertices as [n] and set deg(v) = d[v]
    if type(d) is list:
        d = {v:d[v] for v in range(len(d))}
    #If m is a list, we assume the first value represents m_1
    if type(m) is list:
        m = {(k+1):m[k] for k in range(len(m)) if m[k] > 0}
    #If q is a number, we set q_k = q for all edge sizes k
    if type(q) is float:
        q = {k:q for k in list(m.keys())[1:]}
    #If w is not given, we set w so that all probabilities are equal
    if w is None:
        norm = {k:len([j for j in m.keys() if j<k]) for k in list(m.keys())[1:]}
        w = {}
        for j in list(m.keys())[:-1]:
            w[j] = {}
            for k in m.keys():
                if j<k:
                    w[j][k] = 1/norm[k]

    #If d and m are incompatable, d gets scaled appropriately
    L_d = sum(d.values())
    L_m = sum(k*m[k] for k in m.keys())
    if L_d != L_m:
        scale = L_m/L_d
        for v in d.keys():
            d[v] = round(scale*d[v])
        L_d = sum(d.values())
        while L_d<L_m:
            v = np.random.choice(list(d.keys()))
            d[v] += 1
            L_d += 1
        while L_d>L_m:
            v = np.random.choice(list(d.keys()))
            if d[v]>1:
                d[v] -= 1
                L_d -= 1
    
    return d,m,q,w

def fix_inputs_td(d,m,q,w=None):
    #If d is a list, we label vertices as [n] and set deg(v) = d[v]
    if type(d) is list:
        d = {v:d[v] for v in range(len(d))}
    #If m is a list, we assume the first value represents m_1
    if type(m) is list:
        m = {(k+1):m[k] for k in range(len(m)) if m[k]>0}
    #If q is a number, we set q_k = q for all edge sizes k
    if type(q) is float:
        q = {k:q for k in list(m.keys())[:-1]}
    #If w is not given, we set w so that all probabilities are equal
    if w is None:
        norm = {k:len([l for l in m.keys() if l>k]) for k in list(m.keys())[:-1]}
        w = {}
        for k in list(m.keys())[:-1]:
            w[k] = {}
            for l in m.keys():
                if l>k:
                    w[k][l] = 1/norm[k]

    #If d and m are incompatable, d gets scaled appropriately
    L_d = sum(d.values())
    L_m = sum(k*m[k] for k in m.keys())
    if L_d != L_m:
        scale = L_m/L_d
        for v in d.keys():
            d[v] = round(scale*d[v])
        L_d = sum(d.values())
        while L_d<L_m:
            v = np.random.choice(list(d.keys()))
            d[v] += 1
            L_d += 1
        while L_d>L_m:
            v = np.random.choice(list(d.keys()))
            if d[v]>1:
                d[v] -= 1
                L_d -= 1

    return d,m,q,w

#d: degree sequence
#m: dict of edge sizes
#q: dict of forced-pair probabilities
#w: 2d array of weights
def bottom_up_fast(d,m,q,w=None,multisets=True):
    #Initialize
    d,m,q,w = fix_inputs_bu(d,m,q,w)
    V = list(d.keys())
    L = sum(d.values())
    p = [d[v]/L for v in V]
    E = {k:[] for k in m.keys()}

    #We get a giant list of sampled vertices first and pull from the list as we build
    #This way, we don't call make_edge over and over
    if multisets:
        #If multisets are allowed, we will never pop from big_list more than L times
        big_list = list(np.random.choice(V,size=L,p=p))
    else:
        #If multisets are not allowed, we might pop from big_list more than L times
        #So we initially make the list way too big
        #We also add more stuff to the list if it ever runs out
        big_list = list(np.random.choice(V,size=5*L,p=p))
    
    #Build smallest edges normally
    k_min = list(m.keys())[0]
    for i in range(m[k_min]):
        if multisets:
            e = big_list[-k_min:]
            del big_list[-k_min:]
        else:
            if len(big_list)<k_min:
                big_list = list(np.random.choice(V,size=5*L,p=p))
            e = [big_list.pop()]
            while(len(e)<k_min):
                v = big_list.pop()
                if v not in e:
                    e.append(v)
        E[k_min].append(e)

    #E_list is what gets returned at the end
    E_list = E[k_min].copy()

    #Build bigger edges
    for k in list(m.keys())[1:]:
        #q[k] is the probability of forcing a simplicial edge
        #m[k] is the number of edges of size k
        #so num_simplicial can be generated in advance
        num_simplicial = np.random.binomial(m[k],q[k])

        #Non-simplicial edges are built normally
        for i in range(m[k]-num_simplicial):
            if multisets:
                e = big_list[-k:]
                del big_list[-k:]
            else:
                if len(big_list)<k:
                    big_list = list(np.random.choice(V,size=5*L,p=p))
                e = [big_list.pop()]
                while(len(e)<k):
                    v = big_list.pop()
                    if v not in e:
                        e.append(v)
                    else:
                        if len(big_list)<k:
                            big_list = list(np.random.choice(V,size=5*L,p=p))
            E[k].append(e)

        if num_simplicial > 0:
            #Simplicial edges are built in a particular way
            # We first build a list of all sub-edge sizes we will sample
            sub_sizes = [j for j in m.keys() if j<k]
            weights = [w[j][k] for j in sub_sizes]
            j_list = np.random.choice(sub_sizes,size=num_simplicial,p=weights)
            j_counter = Counter(j_list)
            for j,num_j in j_counter.items():
                #The first part of the edges are previously built, smaller edges
                sub_edge_index_list = np.random.randint(m[j],size=num_j)
                for i in sub_edge_index_list:
                    e = E[j][i].copy()
                    if multisets:
                        e.extend(big_list[-(k-j):])
                        del big_list[-(k-j):]
                    else:
                        if len(big_list)<(k-j):
                            big_list = list(np.random.choice(V,size=5*L,p=p))
                        while(len(e)<k):
                            v = big_list.pop()
                            if v not in e:
                                e.append(v)
                            else:
                                if len(big_list)<k:
                                    big_list = list(np.random.choice(V,size=5*L,p=p))
                    E[k].append(e)
    
        #Add E[k] to E_list
        E_list.extend(E[k].copy())

    return E_list



#d: degree sequence
#m: dict of edge sizes
#q: dict of forced-pair probabilities
#w: 2d array of weights
def top_down_fast(d,m,q,w=None,multisets=True):
    #Initialize
    d,m,q,w = fix_inputs_td(d,m,q,w)
    V = list(d.keys())
    L = sum(d.values())
    p = [d[v]/L for v in V]
    E = {k:[] for k in m.keys()}

    #We get a giant list of sampled vertices first and pull from the list as we build
    #This way, we don't call make_edge over and over
    if multisets:
        #If multisets are allowed, we will never pop from big_list more than L times
        big_list = list(np.random.choice(V,size=L,p=p))
    else:
        #If multisets are not allowed, we might pop from big_list more than L times
        #So we initially make the list way too big
        #We also add more stuff to the list if it ever runs out
        big_list = list(np.random.choice(V,size=5*L,p=p))
    
    #Build largest edges normally
    k_max = list(m.keys())[-1]
    for i in range(m[k_max]):
        if multisets:
            e = big_list[-k_max:]
            del big_list[-k_max:]
        else:
            if len(big_list)<k_max:
                big_list = list(np.random.choice(V,size=5*L,p=p))
            e = [big_list.pop()]
            while(len(e)<k_max):
                v = big_list.pop()
                if v not in e:
                    e.append(v)
        E[k_max].append(e)

    #E_list is what gets returned at the end
    E_list = E[k_max].copy()

    #Build bigger edges
    for k in list(reversed(m.keys()))[1:]:
        #q[k] is the probability of forcing a simplicial edge
        #m[k] is the number of edges of size k
        #so num_simplicial can be generated in advance
        num_simplicial = np.random.binomial(m[k],q[k])

        #Non-simplicial edges are built normally
        for i in range(m[k]-num_simplicial):
            if multisets:
                e = big_list[-k:]
                del big_list[-k:]
            else:
                if len(big_list)<k:
                    big_list = list(np.random.choice(V,size=5*L,p=p))
                e = [big_list.pop()]
                while(len(e)<k):
                    v = big_list.pop()
                    if v not in e:
                        e.append(v)
                    else:
                        if len(big_list)<k:
                            big_list = list(np.random.choice(V,size=5*L,p=p))
            E[k].append(e)

        if num_simplicial > 0:
            #Simplicial edges are built in a particular way
            # We first build a list of all sup-edge sizes we will sample
            sup_sizes = [l for l in m.keys() if l>k]
            weights = [w[k][l] for l in sup_sizes]
            l_list = np.random.choice(sup_sizes,size=num_simplicial,p=weights)
            l_counter = Counter(l_list)
            for l,num_l in l_counter.items():
                #The edge we build is a subset of a previous edge
                sup_edge_index_list = np.random.randint(m[l],size=num_l)
                for i in sup_edge_index_list:
                    e_sup = E[l][i].copy()
                    e = list(random.sample(e_sup,k))
                    E[k].append(e)
    
        #Add E[k] to E_list
        E_list.extend(E[k].copy())

    return E_list






#d: degree sequence
#m: dict of edge sizes
#q: simplicial edge probability
def simplicial_chung_lu_basic(d,m,q):
    #Initialize
    V = list(d.keys())
    L = sum(d.values())
    p = [d[v]/L for v in V]
    E = {k:[] for k in m.keys()}

    #Getting all vertex samples in advance
    vertex_list = list(np.random.choice(V,size=L,p=p))
    
    #Getting the list of edge size samples
    size_list = []
    for k in m.keys():
        size_list.extend([k]*m[k])
    random.shuffle(size_list)

    #Iterate through size_list and build edges
    for k in size_list:
        make_simplicial = (np.random.uniform() < q)
        if make_simplicial:
            #sample_list is the list of edges NOT of size k
            sample_list = []
            for j in m.keys():
                if j != k:
                    sample_list.extend(E[j])
            #if there are no edges to pair with, we build a normal Chung-Lu edge instead
            if len(sample_list) == 0:
                make_simplicial == False
            else:
                sample = np.random.choice(sample_list)
                r = k-len(sample)
                #If the sampled edge is smaller, we build e by joining the sampled edge with vertices from vertex_list
                if r>0:
                    e = sample.copy()
                    e.extend(vertex_list[-r:])
                    vertex_list = vertex_list[:-r]
                #Otherwise, we build e by taking a uniform k-subset of e
                else:
                    e = list(np.random.sample(sample,size=k))
        #Normal chung-lu edges are built here
        if not make_simplicial:
            e = vertex_list[-k:]
            vertex_list = vertex_list[:-k]
        E[k].append(e)
        
    E_all = []
    for k in m.keys():
        E_all.extend(E[k])
    return E_all
        





def normalize(L):
    if type(L) is list:
        norm = sum(L)
        return [x/norm for x in L]
    elif type(L) is dict:
        norm = sum(L.values())
        return {k:v/norm for k,v in L.items()}
    print('invalid input for normalization')
    return


def proc_variables(d, m, q):
    
    #basic fixing
    if type(d) is list:
        d = {v:d[v] for v in range(len(d))}
    #If d is an int, it's assumed to be the desired number of vertices
    #In this case, we give all vertices the same degree
    elif type(d) is int:
        d = {v:1 for v in range(d)}
    V = list(d.keys())
    if type(m) is list:
        m = {(k+1):m[k] for k in range(len(m)) if m[k]>0}
    if (type(q) is float) or (type(q) is int):
        q = {k:q for k in m.keys()}

    #if d and m don't line up, we scale d 
    m_tot = sum(k*m[k] for k in m.keys())
    L = sum(d.values())
    if L != m_tot:
        scale = m_tot/L
        d = {v:round(scale*d[v]) for v in V}
        while sum(d.values()) < m_tot:
            d[np.random.choice(V)] += 1
        while sum(d.values()) > m_tot:
            v = np.random.choice(V)
            if d[v] > 0:
                d[v] -= 1

    #getting the rest of the variables
    L = sum(d.values())
    p = [d[v]/L for v in V]
    E = {k:[] for k in m.keys()}
    
    return d, m, q, V, E, L, p


def get_vertex_list(V, L, p, multisets):
    if multisets:
        return list(np.random.choice(V, size = L, p = p))
    #If multisets are not allowed, we often throw away choices
    #To compensate, we make vertex_list way bigger
    else:
        return list(np.random.choice(V, size = 10*L, p = p))


def get_size_list(m):
    size_list = []
    for k in m.keys():
        size_list.extend([k]*m[k])
    random.shuffle(size_list)
    return size_list


def check_vertex_list(vertex_list, max_edge_size, V, L, p):
    if len(vertex_list) < max_edge_size:
        return get_vertex_list(V, L, p, False)
    else:
        return vertex_list


def make_normal_edge(k, vertex_list, multisets):
    if multisets:
        e = vertex_list[-k:]
        vertex_list = vertex_list[:-k]
    else:
        e = []
        while len(e)<k:
            v = vertex_list.pop()
            if v not in e:
                e.append(v)
    return e, vertex_list


def make_simplicial_edge(k, E, m, vertex_list, multisets):
    #sample_list is the list of edges NOT of size k
    sample_list = []
    for j in m.keys():
        if j != k:
            sample_list.extend(E[j])
    #if there are no edges to pair with, we build a normal Chung-Lu edge instead
    if len(sample_list) == 0:
        return make_normal_edge(k, vertex_list, multisets)
    else:
        sample = sample_list[np.random.randint(len(sample_list))]
        r = k-len(sample)
        #If the sampled edge is smaller, we build e by joining the sampled edge with vertices from vertex_list
        if r>0:
            e = sample.copy()
            if multisets:
                e.extend(vertex_list[-r:])
                vertex_list = vertex_list[:-r]
            else:
                while len(e)<k:
                    v = vertex_list.pop()
                    if v not in e:
                        e.append(v)
        #Otherwise, we build e by taking a uniform k-subset of e
        else:
            e = list(random.sample(sample,k))
        return e, vertex_list


def root(v, parents):
    if parents[v] == v:
        return v
    else:
        return root(parents[v], parents)


def merge(S, parents, sizes):  
    S = {root(v, parents) for v in S}
    if len(S) > 1:
        v = next(iter(S))
        for u in S:
            if sizes[u] > sizes[v]:
                v = u
        for u in S:
            parents[u] = v
        sizes[v] = sum([sizes[u] for u in S])
    return parents, sizes


def get_components(V, parents):
    components = {}
    for v in V:
        rv = root(v, parents)
        if rv in components.keys():
            components[rv].append(v)
        else:
            components[rv] = [v]
    return components


def connect_some(k, d, p, components, sizes, multisets):
    e = []
    root_weights = normalize([sizes[v] for v in components.keys()])
    merge_roots = np.random.choice(list(components.keys()), size = k, p = root_weights, replace = False)
    for v in merge_roots:
        weights = normalize({u:d[u] for u in components[v]})
        u = np.random.choice(components[v], p = list(weights.values()))
        e.append(u)
    return e


def connect_all(V, k, d, p, components, multisets):
    e = []
    for v in components.keys():
        weights = normalize({u:d[u] for u in components[v]})
        u = np.random.choice(components[v], p = list(weights.values()))
        e.append(u)
    excess = k-len(components)
    if excess > 0:
        if multisets:
            e_remaining = list(np.random.choice(V, size = excess, p = p, replace = True))
            e.extend(e_remaining)
        else:
            while len(e) < k:
                v_try = np.random.choice(V, p = p)
                if v_try not in e:
                    e.append(v_try)
    return e
    

def make_skeleton(V, d, m, p, size_list, multisets):
    
    parents = {v : v for v in V}
    sizes = {v:p[i] for i, v in enumerate(V)}
    E = []
    components = {v:[v] for v in V}
    
    while True:
        k = size_list.pop()
        if len(components) < k:
            e = connect_all(V, k, d, p, components, multisets)
            E.append(e)
            break
        else:            
            e = connect_some(k, d, p, components, sizes, multisets)
            E.append(e)
            parents, sizes = merge(e, parents, sizes)            
            components = get_components(V, parents)
    return E, size_list


#d: degree sequence
#m: dict of edge sizes
#q: simplicial edge probability
def simplicial_chung_lu(d, m, q, multisets = True, skeleton = False):
    
    d,m,q,V,E,L,p = proc_variables(d,m,q)
    vertex_list = get_vertex_list(V,L,p,multisets)
    size_list = get_size_list(m)

    #We can optionally create a connected skeleton before continuing
    if skeleton:
        E_all, size_list = make_skeleton(V, d, m, p, size_list, multisets = multisets)
    else:
        E_all = []
    
    #Iterate through size_list and build edges
    for k in size_list:
        make_simplicial = (np.random.uniform() < q[k])
        if make_simplicial:
            e, vertex_list = make_simplicial_edge(k, E, m, vertex_list, multisets)
        #Normal chung-lu edges are built here
        else:
            e, vertex_list = make_normal_edge(k, vertex_list, multisets)
        E[k].append(e)
        if not multisets:
            vertex_list = check_vertex_list(vertex_list, max(m.keys()), V, L, p)
        
    for E_k in E.values():
        E_all.extend(E_k)
    return E_all
