# Simplicial Ratio
Compute simplicial ratio and simplicial matrix of a hypergraph as described in https://arxiv.org/abs/2408.11806

## Simpliciality

In https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-024-00458-1, the authors present three different measures of **simpliciality** that aim to compute how close a hypergraph is to being a simplicial complex.
The code to compute those measures is included in the XGI package (See https://pypi.org/project/xgi/).

We define a **simplicial pair** of edges as two edges of different size where one edge in included in the other.
In https://arxiv.org/abs/2408.11806, we propose a measure where we compute how **surprisingly simplicial** a given hypergraph is when compared to a null model. 
To do so, we define the **simplicial ratio**, which summarises this surprise with a single number by comparing the observed and expected number of simplicial pairs of each sizes.
A simplicial ratio of 1 indicates that the hypergraph has the same simpliciality as the null model, while a value above 1 indicates that the hypergraph is "more simplicial" than expected (and vice-versa for a value below 1).

It can also be useful to look at the matrix **simplicial counts**, where we compute the number of simplicial pairs separately for every pair of edge sizes, as well as the **simplicial matrix**, where we compute the simplicial ratio separately for each pair of edge sizes present in the hypergraph.

The code to compute those new measure can be found in the **simplicial_ratio.py** file, and examples are given in the **simpliciality.ipynb** notebook. Input to the main function is a **list of sets** that enumerates the hyperedges.

### Example

```
code here - TBD
```

Looking at three **contact hypergraphs** used in the papers cited above, we see that the measures reveal different aspects of the data. 
With the simplicial fraction (SF), edit simpliciality (ES) and face edit simpliciality (FES), all values are high, which indicates that all 3 hypergraphs are close to being simplicial complexes.
With the simplicilaity ratio (SR) however, the interpretation is different. The *hospital-lyon* is not surprisingly simplicial while the other two datasets, in particular the *contact-high-school*, are highly "surprisingly" simplicial.
With *hospital-lyon*, there are 1,107 edges of size 2 but only 75 nodes, so the density of 2-edges is very large, and we can expect that a lot of those are subsets of the larger (mainly size 3) edges, thus the lower SR.

```
Results for contact-primary-school
n = 242 
m = 12704 
mean edge size = 2.42
SF:  0.85 
ES:  0.88 
FES: 0.94
SR: 2.68

Results for contact-high-school
n = 327 
m = 7818 
mean edge size = 2.33
SF:  0.81 
ES:  0.91 
FES: 0.92
SR: 6.52

Results for hospital-lyon
n = 75 
m = 1824 
mean edge size = 2.43
SF:  0.91 
ES:  0.94 
FES: 0.97
SR: 0.98
```
