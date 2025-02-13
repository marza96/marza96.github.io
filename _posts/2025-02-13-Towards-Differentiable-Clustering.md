# Towards Differentiable Clustering

## Straight Through Estimator
Hello fellow ML enthusiasts and welcome to my first blog post, where I would like to present an idea, i've stumbled upon while doing research for my Master's Thesis. It is the Straight Through Estimator(STE), first introduced by Hinton et. al. which essentially is an gradient approximation for performing backpropagation through a certain class of nondifferentiable functions

An illustrative example of the STE would be an attempt to backpropagate through a step function which is defined as: 

$$
    h(x) = \begin{cases} 1, & x > 0 \\
                             0, & x \leq 0 \end{cases}
$$
For $h$, the gradient is everywhere except $x=0$ equal to zero and singular at $x=0$. To overcome this issue, Hinton proposes artifficially imposing a gradient function $\dot{h}$ in such cases. A pytorch example showcases this imposition concept, where we consider attempting a backward pass through a step function:
```python
x = torch.tensor((10.0,)).float()
x.requires_grad = True
y = (x > 0) #step function
y.backward() 
```
Executing this will raise a runtime error as there is no grad_fn defined for the step function. The STE ovecromes this issue in the following way:
```python
x = torch.tensor((10.0,)).float()
x.requires_grad = True
y = (x > 0) + x - x.detach() #step function (STE)
y.backward() 
```
In this case, during the forward pass y is evaluated as the step function since the values x and -x.detach() cancel themselves out, but during the backward pass the gradient will flow through x since x.detach() and (x > 0) are detached from the computational graph, effectively imposing an identity gradient on a non-differentiable step function. 

Building on this concept one may think of attempting to solve some discrete optimization problems using this approach. A well known method in the ML comunity is learning permutations using a STE. 
\\TODO insert speech

## K-Means Clustering with STE
K-Means clustering reffers to the problem of assigning points $\mathbf{x}_i$ to a fixed number of clusters $\mathbf{z}_i$ by optimizing: 
$$
    \min_{\mathbf{z}} \sum_i \mathbf{z}_i \|\mathbf{x}_i - \bm{\mu}_i\|_2^2 \\
    \;\;\;\;\;\;\;\;\;\;\;\;\;\; \text{s.t.} \;\; \mathbf{z}_i \in \{0,1\},
$$
where
$$
    \;\;\;\;\;\;\;\;\;\;\;\;\;\; \bm{\mu}_i = \frac{1}{N_i}\sum_i \mathbf{z}_i \mathbf{x}_i 
$$
and 
$$
N_i = \sum_i \mathbf{z}_i.
$$
This problem is usually solved using Expectation Maximization, which is in principle a non-differentiable operation, hence backpropagation through classical K-Means solver is not feasible.

Before describing the method one important concept we need to introduce is the interpretation of K-Means as a decomposition problem. For this purpose we group all the data points in a matrix
$$
    \mathbf{X} = \begin{bmatrix} \mathbf{x}_1^T \\
        \mathbf{x}_2^T \\
        \vdots \\
        \mathbf{x}_n^T \end{bmatrix}
$$
and pose the following decomposition problem:
$$
    \min_{\mathbf{U}, \mathbf{M}} \|\mathbf{X} - \mathbf{U}\mathbf{M}\|_F^2
$$
such that $\mathbf{U} \in \mathbb{R}^{n \times k}$, $\mathbf{k} \in \mathbb{R}^{k \times m}$ and $k < n$. This problem corresponds to finding a low-rank decomposition of $\mathbf{X}$ which minimizes the given frobenius norm of the difference betwen the original and decomposed matrix. Taking the gradient of the bjective gives:
$$

$$


