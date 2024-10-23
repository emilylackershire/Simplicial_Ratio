# Simplicial_Fraction
Compute simplicial fraction and simplicial matrix of a hypergraph as described in https://arxiv.org/abs/2408.11806

## Simpliciality

In https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-024-00458-1, the authors present three different measures of **simpliciality** that aim to compute how close a hypergraph is to being a simplicial complex.
The code to compute those measures is included in the XGI package (See https://pypi.org/project/xgi/).

In https://arxiv.org/abs/2408.11806, we propose a different approach where we compute how **surprisingly simplicial** a given hypergraph is when compared to a null model. To do so, we define two new measures: the **simplicial ratio**, which summarises this surprise with a single number, and the **simplicial matrix**, where we look at this ratio separately for every pair of edge sizes.

A simplicial ratio of 1 indicated that the hypergraph has the same simpliciality as the null model, while values above 1 indicate that the hypergraph is ``more simplicial'' than expected (and vice-versa for values below 1).

The code to compute those new measure can be found in the **simplicial_ratio.py** code, and examples are given in the **simpliciality.ipynb** notebook. 

### Example

```
code here
```
