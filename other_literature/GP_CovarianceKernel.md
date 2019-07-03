## Covariance Kernel

The coregionalized regression model relies on the use of 
$\color{firebrick}{\textbf{multiple output kernels}}$ or $\color{firebrick}{\textbf{vector-valued kernels}}$ of the following form:

$$ 
{\bf B}\otimes{\bf K} =
\left(\begin{array}{ccc}
B_{1,1}\times{\bf K}({\bf X}_{1},{\bf X}_{1}) & \ldots & B_{1,D}\times{\bf K}({\bf X}_{1},{\bf X}_{D})\\
\vdots & \ddots & \vdots\\
B_{D,1}\times{\bf K}({\bf X}_{D},{\bf X}_{1}) & \ldots & B_{D,D}\times{\bf K}({\bf X}_{D},{\bf X}_{D})
\end{array}\right)
$$


In the expression above, ${\bf K}$ is a kernel function, ${\bf B}$ is a regarded as the **coregionalization matrix**, and ${\bf X}_i$ represents the inputs corresponding to the $i$-th output.

Notice that if $B_{i,j} = 0$ for $i \neq j$, then all the outputs are being considered as independent of each other. 

To ensure that the multiple output kernel is a valid kernel, we need the $\bf K$ and ${\bf B}$ to be to be valid. If $\bf K$ is already a valid kernel, we just need to ensure that ${\bf B}$ is positive definite. The last is achieved by defining ${\bf B} = {\bf W}{\bf W}^\top + {\boldsymbol \kappa}{\bf I}$, for some matrix $\bf W$ and vector ${\boldsymbol \kappa}$.

In GPy,a function called $\color{firebrick}ICM$ that deals with the steps of defining two kernels and multiplying them together.
```python
icm = GPy.util.multioutput.ICM(input_dim=1,num_outputs=2,kernel=GPy.kern.RBF(1))
```
