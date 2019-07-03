## Basic Ideas

### Why use S(P)DE solvers for GPs?

- The $O(n^3)$ computational complexity is a challenge.
-  What do we get:
  - $O(n)$ state-space methods for SDEs/SPDEs. 
  - Sparse approximations developed for SPDEs. 
  - Reduced rank Fourier/basis function approximations. Path to non-Gaussian processes.
- Downsides: 
  - We often need to approximate. 
  - Mathematics can become messy

## Stochastic differential equations and Gaussian processes

### Ornstein-Uhlenbeck process

The mean and covariance functions:
$$
\begin{aligned} m(x) &=0 \\ k\left(x, x^{\prime}\right) &=\sigma^{2} \exp \left(-\lambda\left|x-x^{\prime}\right|\right) \end{aligned}
$$
This has a path representation as a stochastic differential equation (SDE):
$$
\frac{d f(t)}{d t}=-\lambda f(t)+w(t)
$$
where $w(t)$ is a white noise process with $x$ relabeled as $t$.

Prove:
$$
\begin{aligned}
FT: (i \omega) \hat{f} &= -\lambda \hat{f} + \hat{\omega} \\
\hat{f} &= \frac{\hat{\omega}}{\lambda +(i \omega) } \\
Spectral Density: \delta(\omega) &= \frac{{E}[|\hat{w}|^{2}]}{w^2+\lambda^2} = \frac{q}{w^2+\lambda^2}\\
IF:h(\tau) &= \frac{1}{2 \pi} \int \frac{q}{w^2+\lambda^2} \exp(iw\tau) d\tau\\
\end{aligned}
$$
Consider a Gaussian process regression problem:
$$
\begin{aligned} f(x) & \sim \mathrm{GP}\left(0, \sigma^{2} \exp \left(-\lambda\left|x-x^{\prime}\right|\right)\right) \\ y_{k} &=f\left(x_{k}\right)+\varepsilon_{k} \end{aligned}
$$
this is equivalent to the state-space model:
$$
\begin{aligned} \frac{d f(t)}{d t} &=-\lambda f(t)+w(t) \\ y_{k} &=f\left(t_{k}\right)+\varepsilon_{k} \end{aligned}
$$
that is, with $fk = f(t_k)$ we have a Gauss-Markov model
$$
\begin{aligned} f_{k+1} & \sim p\left(f_{k+1} | f_{k}\right) \\ y_{k} & \sim p\left(y_{k} | f_{k}\right) \end{aligned}
$$
Solvable in $O(n)$ time using Kalman filter/smoother

![ZZPGcV.png](https://s2.ax1x.com/2019/06/25/ZZPGcV.png)