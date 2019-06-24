# Multiple-Output Gaussian Process

## Working Situation

[![ZEgmxe.png](https://s2.ax1x.com/2019/06/25/ZEgmxe.png)](https://imgchr.com/i/ZEgmxe)

As the picture shows, we want to learn from the three sensors (with complete signal information) to recover the fourth one.

## Dependencies between processes

### Multiple-independent Output GP 

![ZEgMqA.png](https://s2.ax1x.com/2019/06/25/ZEgMqA.png)
$$
\begin{aligned} f_{1}(\mathbf{x}) \sim \mathcal{G} \mathcal{P}\left(0, k_{1}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right) &\;\;\;\;\;\;\;\;\;\; f_{2}(\mathbf{x}) \sim \mathcal{G} \mathcal{P}\left(0, k_{2}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right) \\ D_{1}=\left\{\left(\mathbf{x}_{i, 1}, y_{1}\left(\mathbf{x}_{i, 2}\right)\right) | i=1, \ldots, N_{1}\right\} & \;\;\;\;\;\;\;\;\;\;\mathcal{D}_{2}=\left\{\left(\mathbf{x}_{i, 2}, y_{2}\left(\mathbf{x}_{i, 2}\right)\right) | i=1, \ldots, N_{2}\right\} \\ \mathbf{y}_{1} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{K}_{1}+\sigma_{1}^{2}\right) & \;\;\;\;\;\;\;\;\;\;\mathbf{y}_{2} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{K}_{2}+\sigma_{2}^{2} \mathbf{l}\right) \end{aligned}
$$

$$
\left[\begin{array}{l}{\mathbf{y}_{1}} \\ {\mathbf{y}_{2}}\end{array}\right] \sim \mathcal{N}\left(\left[\begin{array}{l}{\mathbf{0}} \\ {\mathbf{0}}\end{array}\right],\left[\begin{array}{cc}{\mathbf{K}_{1}} & {\mathbf{0}} \\ {\mathbf{0}} & {\mathbf{K}_{2}}\end{array}\right]+\left[\begin{array}{cc}{\sigma_{1}^{2} \mathbf{l}} & {\mathbf{0}} \\ {\mathbf{0}} & {\sigma_{2}^{2} \mathbf{l}}\end{array}\right]\right)
$$

### How to find the independences for kernel design

$$
\mathbf{K}_{\mathbf{f}, \mathbf{f}}=\left[\begin{array}{cc}{\mathbf{K}_{1}} & {?} \\ {?} & {\mathbf{K}_{2}}\end{array}\right]
$$

Build a cross-covariance function $cov[f_1(x), f_2(x^{'})]$ such that $K_{f,f}$ is positive semi-definite.

### Different input configurations of  data

[![ZEcz2F.md.png](https://s2.ax1x.com/2019/06/25/ZEcz2F.md.png)](https://imgchr.com/i/ZEcz2F)
$$
\begin{array}{ll}{\mathcal{D}_{1}=\left\{\left(\mathbf{x}_{i}, f_{1}\left(\mathbf{x}_{i}\right)\right)_{i=1}^{N}\right\}} &\;\;\;\;\; {\mathcal{D}_{1}=\left\{\left(\mathbf{x}_{i, 1}, f_{1}\left(\mathbf{x}_{i, 1}\right)\right)_{i=1}^{N_{1}}\right\}} \\ {\mathcal{D}_{2}=\left\{\left(\mathbf{x}_{i}, f_{2}\left(\mathbf{x}_{i}\right)\right)_{i=1}^{N}\right\}} & \;\;\;\;\;{\mathcal{D}_{2}=\left\{\left(\mathbf{x}_{i, 2}, f_{2}\left(\mathbf{x}_{i, 2}\right)\right)_{i=1}^{N_{2}}\right\}}\end{array}
$$

## Intrinsic Coregionalization Model

### Two outputs

#### Sample Once

Consider two outputs $f_1(x) $$f_{2}(x)$ with $x\in \mathcal{R}^{p}$. 

1. Sample from a GP $u(\mathbf{x}) \sim \mathcal{G P}\left(0, k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right)$ to obtain $u^{1}(\mathbf{x})$
2. Obtain  $f_1(x) $ and $f_{2}(x)$ by linearly transforming:

$$
\begin{aligned} f_{1}(\mathbf{x}) &=a_{1}^{1} u^{1}(\mathbf{x}) \\ f_{2}(\mathbf{x}) &=a_{2}^{1} u^{1}(\mathbf{x}) \end{aligned}
$$

For a fixed value $x$.  we can group $f_1(x)$ and $f_2(x)$ in a vector:
$$
\mathbf{f}(\mathbf{x})=\left[\begin{array}{l}{f_{1}(\mathbf{x})} \\ {f_{2}(\mathbf{x})}\end{array}\right]
$$
and this vector will be refer as a $\bf{vector-valued \; function}$.

The covariance for $f(x)$ is computed as:
$$
\operatorname{cov}\left(\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right)=\mathbb{E}\left\{\mathbf{f}(\mathbf{x})\left[\mathbf{f}\left(\mathbf{x}^{\prime}\right)\right]^{\top}\right\}-\mathbb{E}\{\mathbf{f}(\mathbf{x})\}\left[\mathbb{E}\left\{\mathbf{f}\left(\mathbf{x}^{\prime}\right)\right\}\right]^{\top}
$$

$$
\mathbb{E}\left\{\left[\begin{array}{c}{f_{1}(\mathbf{x})} \\ {f_{2}(\mathbf{x})}\end{array}\right]\left[\begin{array}{ll}{f_{1}\left(\mathbf{x}^{\prime}\right)} & {f_{2}\left(\mathbf{x}^{\prime}\right) ]}\end{array}\right\}=\left[\begin{array}{cc}{\mathbb{E}\left\{f_{1}(\mathbf{x}) f_{1}\left(\mathbf{x}^{\prime}\right)\right\}} & {\mathbb{E}\left\{f_{1}(\mathbf{x}) f_{2}\left(\mathbf{x}^{\prime}\right)\right\}} \\ {\mathbb{E}\left\{f_{2}(\mathbf{x}) f_{1}\left(\mathbf{x}^{\prime}\right)\right\}} & {\mathbb{E}\left\{f_{2}(\mathbf{x}) f_{2}\left(\mathbf{x}^{\prime}\right)\right\}}\end{array}\right]\right.\\
\begin{aligned} \mathbb{E}\left\{f_{1}(\mathbf{x}) f_{1}\left(\mathbf{x}^{\prime}\right)\right\} &=\mathbb{E}\left\{a_{1}^{1} u^{1}(\mathbf{x}) a_{1}^{1} u^{1}\left(\mathbf{x}^{\prime}\right)\right\}=\left(a_{1}^{1}\right)^{2} \mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\} \\ \mathbb{E}\left\{f_{1}(\mathbf{x}) f_{2}\left(\mathbf{x}^{\prime}\right)\right\} &=\mathbb{E}\left\{a_{1}^{1} u^{1}(\mathbf{x}) a_{2}^{1}\left(\mathbf{x}^{\prime}\right)\right\}=a_{1}^{1} a_{2}^{1} \mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\} \\ \mathbb{E}\left\{f_{2}(\mathbf{x}) f_{2}\left(\mathbf{x}^{\prime}\right)\right\} &=\mathbb{E}\left\{a_{2}^{1} u^{1}(\mathbf{x}) a_{2}^{1} u^{1}\left(\mathbf{x}^{\prime}\right)\right\}=\left(a_{2}^{1}\right)^{2} \mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\} \end{aligned}
$$

So that term could be written as:
$$
\mathbb{E}\left\{\mathbf{f}(\mathbf{x})\left[\mathbf{f}\left(\mathbf{x}^{\prime}\right)\right]^{\top}\right\} =\left[\begin{array}{cc}{\left(a_{1}^{1}\right)^{2} \mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\}} & {a_{1}^{1} a_{2}^{1} \mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\}} \\ {a^{1} a^{2} \mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\}} & {\left(a_{2}^{1}\right)^{2} \mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\}}\end{array}\right]\\
=\left[\begin{array}{cc}{\left(a_{1}^{1}\right)^{2}} & {a_{1}^{1} a_{2}^{1}} \\{a_{1}^{1} a_{2}^{1}} & {\left(a_{2}^{1}\right)^{2}}\end{array}\right] \mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\}
$$
The term $\mathbb{E}\{\mathbf{f}(\mathbf{x})\}$ is computed as:
$$
\mathbb{E}\left\{\left[\begin{array}{c}{f_{1}(\mathbf{x})} \\ {f_{2}(\mathbf{x})}\end{array}\right]\right\}=\left[\begin{array}{c}{\mathbb{E}\left\{f_{1}(\mathbf{x})\right\}} \\ {\mathbb{E}\left\{f_{1}(\mathbf{x})\right\}}\end{array}\right]=\left[\begin{array}{c}{\mathbb{E}\left\{a_{1}^{1} u^{1}(\mathbf{x})\right\}} \\ {\mathbb{E}\left\{a_{2}^{1} u^{1}(\mathbf{x})\right\}}\end{array}\right] ]=\left[\begin{array}{c}{a_{1}^{1}} \\ {a_{2}^{1}}\end{array}\right] \mathbb{E}\left\{u^{1}(\mathbf{x})\right\}
$$
Putting them together, the covariance for $f(x^{'})$ follows as:
$$
\left[\begin{array}{cc}{\left(a_{1}^{1}\right)^{2}} & {a_{1}^{1} a_{2}^{1}} \\ {a_{1}^{1} a_{2}^{1}} & {\left(a_{2}^{1}\right)^{2}}\end{array}\right] \mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\}-\left[\begin{array}{c}{a_{1}^{1}} \\ {a_{2}^{1}}\end{array}\right]\left[\begin{array}{cc}{a_{1}^{1}} & {a_{2}^{1} ]}\end{array}\left\{u^{1}(\mathbf{x})\right\} \mathbb{E}\left\{u^{1}\left(\mathbf{x}^{\prime}\right)\right\}\right.
$$
Defining $\mathbf{a}=\left[\begin{array}{ll}{a_{1}^{1}} & {a_{2}^{1}}\end{array}\right]^{\top}$,
$$
\begin{aligned} \operatorname{cov}\left(\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right) &=\mathbf{a a}^{\top} \mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\}-\mathbf{a a}^{\top} \mathbb{E}\left\{u^{1}(\mathbf{x})\right\} \mathbb{E}\left\{u^{1}\left(\mathbf{x}^{\prime}\right)\right\} \\ &=\mathbf{a a}^{\top} \underbrace{\left[\mathbb{E}\left\{u^{1}(\mathbf{x}) u^{1}\left(\mathbf{x}^{\prime}\right)\right\}-\mathbb{E}\left\{u^{1}(\mathbf{x})\right\} \mathbb{E}\left\{u^{1}\left(\mathbf{x}^{\prime}\right)\right\}\right]}_{k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)} \\ &=\mathbf{a} \mathbf{a}^{\top} k\left(\mathbf{x}, \mathbf{x}^{\prime}\right) \end{aligned}
$$
We define $\mathbf{B}=\mathbf{a a}^{\top}$, leading to
$$
\operatorname{cov}\left(\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right)=\mathbf{B} k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\left[\begin{array}{ll}{b_{11}} & {b_{12}} \\ {b_{21}} & {b_{22}}\end{array}\right] k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)
$$
and the $\bf{B}$ has rank one, since it is the result of the multiplication of two column-vector.



#### Sample Twice

Sample **twice** from a GP $u(\mathbf{x}) \sim \mathcal{G} \mathcal{P}\left(0, k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right)$ to obtain $u^{1}(\mathbf{x}) \text { and } u^{2}(\mathbf{x})$. 

Adding a scaled transformation.:
$$
\begin{array}{l}{f_{1}(\mathbf{x})=a_{1}^{1} u^{1}(\mathbf{x})+a_{1}^{2} u^{2}(\mathbf{x})} \\ {f_{2}(\mathbf{x})=a_{2}^{1} u^{1}(\mathbf{x})+a_{2}^{2} u^{2}(\mathbf{x})}\end{array}
$$
[![ZEcx8U.md.png](https://s2.ax1x.com/2019/06/25/ZEcx8U.md.png)](https://imgchr.com/i/ZEcx8U)**

Notice that the $u_1$ and $u_2$ are independent, although they share the same covariance $k$.
$$
\mathbf{f}(\mathbf{x}) = \left[\begin{array}{cc}{\left(a_{1}^{1}\right)^{}} & {a_{1}^{2} } \\ {a_{2}^{1} } & {\left(a_{2}^{2}\right)^{}}\end{array}\right] \left[\begin{array}{l}{u^{1}} \\ {u^{2}}\end{array}\right]
$$
The vector-valued function can be written as $f(x)$, where $\mathbf{a}^{1}=\left[a_{1}^{1 } \;\;a_{2}^{1}\right]^{\top} \text { and } \mathbf{a}^{2}=\left[a_{1}^{2}\;\; a_{2}^{2}\right]^{\top}$

The covariance for $f(x)$ is computed as:
$$
\begin{aligned} \operatorname{cov}\left(\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right) &=\mathbf{a}^{1}\left(\mathbf{a}^{1}\right)^{\top} \operatorname{cov}\left(u^{1}(\mathbf{x}), u^{1}\left(\mathbf{x}^{\prime}\right)\right)+\mathbf{a}^{2}\left(\mathbf{a}^{2}\right)^{\top} \operatorname{cov}\left(u^{2}(\mathbf{x}), u^{2}\left(\mathbf{x}^{\prime}\right)\right) \\ &=\mathbf{a}^{1}\left(\mathbf{a}^{1}\right)^{\top} k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)+\mathbf{a}^{2}\left(\mathbf{a}^{2}\right)^{\top} k\left(\mathbf{x}, \mathbf{x}^{\prime}\right) \\ &=\left[\mathbf{a}^{1}\left(\mathbf{a}^{1}\right)^{\top}+\mathbf{a}^{2}\left(\mathbf{a}^{2}\right)^{\top}\right] k\left(\mathbf{x}, \mathbf{x}^{\prime}\right) \end{aligned}
$$
notice that  $u_1$ and $u_2$ are independent, so their variance could be added directly.

we define $\mathbf{B}=\mathbf{a}^{1}\left(\mathbf{a}^{1}\right)^{\top}+\mathbf{a}^{2}\left(\mathbf{a}^{2}\right)^{\top}$, leading to:
$$
\operatorname{cov}\left(\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right)=\mathbf{B} k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\left[\begin{array}{ll}{b_{11}} & {b_{12}} \\ {b_{21}} & {b_{22}}\end{array}\right] k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)
$$
Notice that $B$ has rank two.

Observed Data:

[![ZEg9KJ.md.png](https://s2.ax1x.com/2019/06/25/ZEg9KJ.md.png)](https://imgchr.com/i/ZEg9KJ)
$$
\left[\begin{array}{c}{\mathbf{f}_{1}} \\ {\mathbf{f}_{2}}\end{array}\right]=\left[\begin{array}{c}{f_{1}\left(\mathbf{x}_{1}\right)} \\ {\vdots} \\ {f_{1}\left(\mathbf{x}_{N}\right)} \\ {f_{2}\left(\mathbf{x}_{1}\right)} \\ {\vdots} \\ {f_{2}\left(\mathbf{x}_{N}\right)}\end{array}\right] \sim \mathcal{N}\left(\left[\begin{array}{l}{\mathbf{0}} \\ {\mathbf{0}}\end{array}\right],\left[\begin{array}{cc}{b_{11} \mathbf{K}} & {b_{12} \mathbf{K}} \\ {b_{21} \mathbf{K}} & {b_{22} \mathbf{K}}\end{array}\right]\right)
$$
The matrix $\bf{k} \in \mathcal{R}^{N*N}$ has elements $k(x_i,x_j)$.

If we use **Kronecker product** we would get:


$$
\left[\begin{array}{c}{\mathbf{f}_{1}} \\ {\mathbf{f}_{2}}\end{array}\right]=\left[\begin{array}{c}{f_{1}\left(\mathbf{x}_{1}\right)} \\ {\vdots} \\ {f_{1}\left(\mathbf{x}_{N}\right)} \\ {f_{2}\left(\mathbf{x}_{1}\right)} \\ {\vdots} \\ {f_{2}\left(\mathbf{x}_{N}\right)}\end{array}\right] \sim \mathcal{N}\left(\left[\begin{array}{l}{\mathbf{0}} \\ {\mathbf{0}}\end{array}\right], \mathbf{B} \otimes \mathbf{K}\right)
$$

### General Case

Consider a set of functions $\left\{f_{d}(\mathbf{x})\right\}_{d=1}^{D}$.

In the ICM,
$$
f_{d}(\mathbf{x})=\sum_{i=1}^{R} a_{d}^{i} u^{i}(\mathbf{x})
$$
where the functions $u_i(x)$ are GPs sampled independently, and share the same covariance function $k(x, x^{'})$.

For $\mathbf{f}(\mathbf{x})=\left[f_{1}(\mathbf{x}) \cdots f_{D}(\mathbf{x})\right]^{\top}$, the covariance is given as:
$$
\operatorname{cov}\left[\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right]=\mathbf{A} \mathbf{A}^{\top} k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\mathbf{B} k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)
$$
where 
$$
\mathbf{A}=\left[\mathbf{a}^{1} \mathbf{a}^{2} \cdots \mathbf{a}^{R}\right]
$$
and the Rank of $B$ is given by $R$.

### ICM: autokrigeability

If the outputs are considered to be noise-free, prediction using the ICM under an isotopic data case is equivalent to independent prediction over each output. This circumstance is also known as autokrigeability.

The prove:

Assume that we only have two outputs: $f_1,f_2$

the predicated mean could be written as:
$$
\mu = K_{f_{*},f} (K_{f,f})^{-1}f\\
K_{f,f} = B \otimes K
$$

$$
\begin{aligned} \mu &= B \otimes K_{*} (B \otimes K)^{-1} f\\
&= B \otimes K_{*}  (B^{-1} \otimes K^{-1})f\\
&= BB^{-1}\otimes K_{*}K^{-1}f\\
&=I \otimes K_{*}K^{-1}f \\
&=\begin{bmatrix}
K_{*}K^{-1} &  0\\ 
0 & K_{*}K^{-1}
\end{bmatrix}\begin{bmatrix}
f_{1}\\ 
f_{2}
\end{bmatrix}\end{aligned}
$$

it means, the prediction of $f_{1}$ only depends on the data set for $f_{1}$

## Semiparametric Latent Factor Model (SLFM)

ICM uses R samples $u^{i}(x)$ from $u(x)$ with the same covariance function. SLFM uses Q samples from $u_{q}$ processes with different covariance functions.

### Two Outputs

1. Sample from a GP $\mathcal{G P}\left(0, k_{1}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right)$ to obtain $u_1(x)$.

2. Sample from a GP $\mathcal{G P}\left(0, k_{2}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right)$ to obtain $u_2(x)$.

3. Adding a scaled versions:
   $$
   \begin{array}{l}{f_{1}(\mathbf{x})=a_{1,1} u_{1}(\mathbf{x})+a_{1,2} u_{2}(\mathbf{x})} \\ {f_{2}(\mathbf{x})=a_{2,1} u_{1}(\mathbf{x})+a_{2,2} u_{2}(\mathbf{x})}\end{array}
   $$
   

[![ZEcvCT.md.png](https://s2.ax1x.com/2019/06/25/ZEcvCT.md.png)](https://imgchr.com/i/ZEcvCT)

Similar, it can be written as:
$$
\mathbf{f}(\mathbf{x})=\mathbf{a}_{1} u_{1}(\mathbf{x})+\mathbf{a}_{2} u_{2}(\mathbf{x})
$$
with $\mathbf{a}_{1}=\left[a_{1,1} a_{2,1}\right]^{\top} \text { and } \mathbf{a}_{2}=\left[a_{1,2} a_{2,2}\right]^{\top}$

The covariance for $f(x)$ is computed as:
$$
\begin{aligned} \operatorname{cov}\left(\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right) &=\mathbf{a}_{1}\left(\mathbf{a}_{1}\right)^{\top} \operatorname{cov}\left(u_{1}(\mathbf{x}), u_{1}\left(\mathbf{x}^{\prime}\right)\right)+\mathbf{a}_{2}\left(\mathbf{a}_{2}\right)^{\top} \operatorname{cov}\left(u_{2}(\mathbf{x}), u_{2}\left(\mathbf{x}^{\prime}\right)\right) \\ &=\mathbf{a}_{1}\left(\mathbf{a}_{1}\right)^{\top} k_{1}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)+\mathbf{a}_{2}\left(\mathbf{a}_{2}\right)^{\top} k_{2}\left(\mathbf{x}, \mathbf{x}^{\prime}\right) \end{aligned}
$$
We define $\mathbf{B}_{1}=\mathbf{a}_{1}\left(\mathbf{a}_{1}\right)^{\top} \text { and } \mathbf{B}_{2}=\mathbf{a}_{2}\left(\mathbf{a}_{2}\right)^{\top}$, leading to:
$$
\operatorname{cov}\left(\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right)=\mathbf{B}_{1} k_{1}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)+\mathbf{B}_{2} k_{2}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)
$$
Notice that $B_{1} $ and $B_{2}$ have rank one.

[![ZEcX5V.png](https://s2.ax1x.com/2019/06/25/ZEcX5V.png)](https://imgchr.com/i/ZEcX5V)
$$
\left[\begin{array}{c}{\mathbf{f}_{1}} \\ {\mathbf{f}_{2}}\end{array}\right]=\left[\begin{array}{c}{f_{1}\left(\mathbf{x}_{1}\right)} \\ {\vdots} \\ {f_{1}\left(\mathbf{x}_{N}\right)} \\ {f_{2}\left(\mathbf{x}_{1}\right)} \\ {\vdots} \\ {f_{2}\left(\mathbf{x}_{N}\right)}\end{array}\right] \sim \mathcal{N}\left(\left[\begin{array}{l}{\mathbf{0}} \\ {\mathbf{0}}\end{array}\right], \mathbf{B}_{1} \otimes \mathbf{K}_{1}+\mathbf{B}_{2} \otimes \mathbf{K}_{2}\right)
$$

### General Case:

Consider a set of functions $\left\{f_{d}(\mathbf{x})\right\}_{d=1}^{D}$

In the SLFM,
$$
f_{d}(\mathbf{x})=\sum_{q=1}^{Q} a_{d, q} u_{q}(\mathbf{x})
$$
where the functions $u_{q}(x)$ are GPs with covariance functions $k_{q}(x,x^{'})$.

For $\mathbf{f}(\mathbf{x})=\left[f_{1}(\mathbf{x}) \cdots f_{D}(\mathbf{x})\right]^{\top}$, the covariance is given as:
$$
\operatorname{cov}\left[\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right]=\sum_{q=1}^{Q} \mathbf{A}_{q} \mathbf{A}_{q}^{\top} k_{q}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\sum_{q=1}^{Q} \mathbf{B}_{q} k_{q}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)
$$
where $A_{q} = a_{q}$.

The rank of each $B_{q}$ is one.

## Linear model of coregionalization (LMC)

The LMC generalizes the ICM and the SLFM allowing several independent samples from GPs with different covariances.

Consider a set of functions $\left\{f_{d}(\mathbf{x})\right\}_{d=1}^{D}$
$$
f_{d}(\mathbf{x})=\sum_{q=1}^{Q} \sum_{i=1}^{R_{q}} a_{d, q}^{i} u_{q}^{i}(\mathbf{x})
$$
where the functions $u_{q}^{i}$ are GPs with zero means and covariance functions:
$$
\operatorname{cov}\left[u_{q}^{i}(\mathbf{x}), u_{q^{\prime}}^{i^{\prime}}\left(\mathbf{x}^{\prime}\right)\right]=k_{q}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)
$$
if $i = i^{'}$ and $q = q^{'}$

There are $Q$ groups of samples. For each group, there are $R_{q}$ samples obtained independently from the same GP with covariance $k_q(x,x^{'})$.

[![ZEgSv4.md.png](https://s2.ax1x.com/2019/06/25/ZEgSv4.md.png)](https://imgchr.com/i/ZEgSv4)

The LMC corresponds to the sum of Q ICMs.

Suppose we have D = 2, Q = 2, and $R_q$=2. According to LMC:
$$
\begin{array}{l}{f_{1}(\mathbf{x})=a_{1,1}^{1} u_{1}^{1}(\mathbf{x})+a_{1,1}^{2} u_{1}^{2}(\mathbf{x})+a_{1,2}^{1} u_{2}^{1}(\mathbf{x})+a_{1,2}^{2} u_{2}^{2}(\mathbf{x})} \\ {f_{2}(\mathbf{x})=a_{2,1}^{1} u_{1}^{1}(\mathbf{x})+a_{2,1}^{2} u_{1}^{2}(\mathbf{x})+a_{2,2}^{1} u_{2}^{1}(\mathbf{x})+a_{2,2}^{2} u_{2}^{2}(\mathbf{x})}\end{array}
$$
For $\mathbf{f}(\mathbf{x})=\left[f_{1}(\mathbf{x}) \cdots f_{D}(\mathbf{x})\right]^{\top}$, the covariance $\operatorname{cov}\left[\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right]$ is given as:
$$
\operatorname{cov}\left[\mathbf{f}(\mathbf{x}), \mathbf{f}\left(\mathbf{x}^{\prime}\right)\right]=\sum_{q=1}^{Q} \mathbf{A}_{q} \mathbf{A}_{q}^{\top} k_{q}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\sum_{q=1}^{Q} \mathbf{B}_{q} k_{q}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)
$$
where $\mathbf{A}_{q}=\left[\mathbf{a}_{q}^{1} \mathbf{a}_{q}^{2} \cdots \mathbf{a}_{q}^{R_{q}}\right]$.

The rank of each $B_{q}$ is $R_{q}$. 

The matrices $B_{q}$ are known as the coregionalization matrices.

[![ZEgCr9.md.png](https://s2.ax1x.com/2019/06/25/ZEgCr9.md.png)](https://imgchr.com/i/ZEgCr9)
$$
\left[\begin{array}{c}{\mathbf{f}_{1}} \\ {\mathbf{f}_{2}}\end{array}\right]=\left[\begin{array}{c}{f_{1}\left(\mathbf{x}_{1}\right)} \\ {\vdots} \\ {f_{1}\left(\mathbf{x}_{N}\right)} \\ {f_{2}\left(\mathbf{x}_{1}\right)} \\ {\vdots} \\ {f_{2}\left(\mathbf{x}_{N}\right)}\end{array}\right] \sim \mathcal{N}\left(\left[\begin{array}{l}{\mathbf{0}} \\ {\mathbf{0}}\end{array}\right], \sum_{q=1}^{Q} \mathbf{B}_{q} \otimes \mathbf{K}_{q}\right)
$$
