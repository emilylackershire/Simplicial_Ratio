import numpy as np
from scipy.special import binom
from itertools import chain, combinations

def truncated_power_law(n, minval, maxval, r):
    u = np.random.random(n)
    a = minval ** (1 - r)
    b = maxval ** (1 - r)
    return {i: int(val) for i, val in enumerate((a + u * (b - a)) ** (1 / (1 - r)))}


def truncated_power_law_mean(minval, maxval, r):
    a = minval ** (2 - r)
    b = maxval ** (2 - r)
    c = minval ** (1 - r)
    d = maxval ** (1 - r)
    return (r - 1) / (r - 2) * (a - b) / (c - d)


def list_of_lists_to_latex_table(data, column_labels, decimals=2):
    """list of lists and column labels to latex table

    Parameters
    ----------
    data : list of lists
        list of table rows
    column_labels : list of str
        columns labels

    Raises
    ------
    Exception
        If the dimensions don't match.
    """

    if len(data[0]) != len(column_labels):
        raise Exception("Column labels and data entries must have the same length!")

    m = len(column_labels)
    print("\\begin{table}")
    print("\\begin{center}")
    print("\\begin{tabular}{" + "c" * m + "}")
    print(" & ".join([str(e) for e in column_labels]) + " \\\ ")
    print("\\hline")
    for row in data:
        print(
            " & ".join(
                [str(round(e, decimals) if isinstance(e, float) else e) for e in row]
            )
            + " \\\ "
        )
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")

# This implements the size-restricted power set
def powerset(iterable, min_size=1, max_size=None):
    """Generates a modified powerset.

    User can specify the maximum and minimum size
    of the sets in the powerset.

    Parameters
    ----------
    iterable : iterable
        The set for which to compute the powerset.
    min_size: int, default: 1
        The minimum hyperedge size to include when
        calculating whether a hyperedge is a simplex
        by counting subfaces.
    max_size : int, default: None.
        The maximum size to include when computing
        the power set. When max_size=None, it generates
        the powerset including the edge itself.

    Returns
    -------
    itertools.chain
        a generator of the sets in the powerset.
    """
    s = iterable
    if max_size is None:
        max_size = len(s)

    return chain.from_iterable(
        combinations(s, r) for r in range(min_size, max_size + 1)
    )


def count_subfaces(t, face, min_size=1):
    """Computing the edit distance for a single face.

    Parameters
    ----------
    t : Trie
        The trie representing the hypergraph
    face : iterable
        The edge for which to find the edit distance
    min_size: int, default: 1
        The minimum hyperedge size to include when
        calculating whether a hyperedge is a simplex
        by counting subfaces.

    Returns
    -------
    int
        The edit distance
    """
    sub_edges = list(powerset(face, min_size=min_size, max_size=len(face) - 1))
    count = 0
    for e in sub_edges:
        if t.search(e):
            count += 1

    return count


def missing_subfaces(t, face, min_size=1):
    """Computing the edit distance for a single face.

    Parameters
    ----------
    t : Trie
        The trie representing the hypergraph
    face : iterable
        The edge for which to find the edit distance
    min_size: int, default: 1
        The minimum hyperedge size to include when
        calculating whether a hyperedge is a simplex
        by counting subfaces.

    Returns
    -------
    int
        The edit distance
    """
    sub_edges = list(powerset(face, min_size=min_size, max_size=len(face) - 1))
    ms = set()
    for e in sub_edges:
        if not t.search(e):
            ms.add(frozenset(e))
    return ms


def max_number_of_subfaces(min_size, max_size):
    d = 2**max_size - 2  # subtract 2 for the face itself and the empty set
    for i in range(1, min_size):
        d -= binom(max_size, i)
    return int(d)


def simplicial_assortativity(H, metric, weighted=False):
    if metric == "sf":
        s = H.nodes.local_simplicial_fraction.asnumpy()
    elif metric == "es":
        s = H.nodes.local_edit_simpliciality.asnumpy()
    elif metric == "fes":
        s = H.nodes.local_face_edit_simpliciality.asnumpy()
    else:
        raise Exception(f"{metric} is an invalid metric!")

    A = xgi.adjacency_matrix(H, sparse=False, weighted=True)
    n = A.shape[0]
    x = []
    y = []
    for i in range(n):
        for j in range(i):
            if A[i, j] and not np.isnan(s[i]) and not np.isnan(s[j]):
                if weighted:
                    x.extend([s[i]] * int(A[i, j]))
                    x.extend([s[j]] * int(A[i, j]))
                    y.extend([s[j]] * int(A[i, j]))
                    y.extend([s[i]] * int(A[i, j]))
                else:
                    x.append(s[i])
                    x.append(s[j])
                    y.append(s[j])
                    y.append(s[i])
    return np.corrcoef(x, y)[0, 1]
