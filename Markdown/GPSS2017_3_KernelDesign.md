## What is a kernel?

Theorem (Leove):

$k$ corresponds to the covariance of a GP.

$k$ is a symmetric positive semi-definite function.

when $k$ is a function of $x-y$ , the kernel is called stationary, $\sigma$ is called the variance and $\theta$ is called the lengthscale. And it is quite import to look at the length scale after the optimization step, if the value is quite small that means your model will gain no information from the surrounding, and thus no good prediction for the testing points.

## Choosing appropriate kernel

In order to choose a kernel, one should gather all possible
information about the function to approximate:

- Is it stationary ?
- Is it differentiable, what’s its regularity ?
- Do we expect particular trends ?
- Do we expect particular patterns (periodicity, cycles, additivity) ?

Kernels often include rescaling parameters : θ for the x axis (length-scale) and σ for the y ($\sigma^{2}$  often corresponds to the GP variance). They can be tuned by

- maximizing the likelihood
- minimizing the prediction error

It is common to try various kernels and to asses the model accuracy. The idea is to compare some model predictions against actual values :

- On a test set
- Using leave-one-out

Furthermore, it is often interesting to try some input remapping such as $x \rightarrow \log (x), x \rightarrow \exp (x)$ to make our data set stationary, and then choose to use the stationary kernel.

## Making new from old

- Summed together
  - On the same space k(x, y) = k1(x, y) + k2(x, y)
  - On the tensor space k(x, y) = k1(x1, y1) + k2(x2, y2)
- Multiplied together
  - On the same space k(x, y) = k1(x, y) × k2(x, y)
  - On the tensor space k(x, y) = k1(x1, y1) × k2(x2, y2)
- Composed with a function
  -  k(x, y) = k1(f (x), f (y))

### Example: $CO_2$

![1560890999880](C:\Users\13269\Desktop\python_Markdown\GPSS\Notebook-for-GPSS2017\pics\02-06.PNG)

- First, we consider a squared-exponential kernel:
  $$
  k(x, y)=\sigma^{2} \exp \left(-\frac{(x-y)^{2}}{\theta^{2}}\right)
  $$
  

![1560891091853](C:\Users\13269\Desktop\python_Markdown\GPSS\Notebook-for-GPSS2017\pics\02-07.PNG)

First, we would like to say that we observe the high frequency in the data, so we would like to choose a very small length scale value, the result is shown as the left picture. The reason is that, you choose the small scale value, when you are in 2020, the model will get no information from the data set and not influenced by the past values.

Second choice is to be focus on the trend, with low frequency which would lead to a very large length scale value, as shown in picture right. But the confidence interval is over confident.

- Second, we sum both kernels together.
  $$
  k(x, y)=k_{r b f 1}(x, y)+k_{r b f 2}(x, y)
  $$
  ![1560891556775](C:\Users\13269\Desktop\python_Markdown\GPSS\Notebook-for-GPSS2017\pics\02-08.PNG)

One thing to notice that, even in the second choice (combination choice), it seems that we would have more parameters, but indeed the optimization process will get easier than the first choice. Because for the first two model, the likelihood will get very very small, since both assumptions make sense for data we have.

- Then, adding the periodic term into the kernel. 
  $$
  k(x, y)=\sigma_{0}^{2} x^{2} y^{2}+k_{r b f 1}(x, y)+k_{r b f 2}(x, y)+k_{p e r}(x, y)
  $$
  

![1560891925621](C:\Users\13269\Desktop\python_Markdown\GPSS\Notebook-for-GPSS2017\pics\02-09.PNG)

###  Sum of  kernels  over  tensor  space

Property:
$$
k(\mathbf{x}, \mathbf{y})=k_{1}\left(x_{1}, y_{1}\right)+k_{2}\left(x_{2}, y_{2}\right)
$$
is a valid covariance structure.

![1560893886801](C:\Users\13269\Desktop\python_Markdown\GPSS\Notebook-for-GPSS2017\pics\02-10.PNG)

Tensor Additive kernels are very useful for:

- Approximating additive functions
- Building models over high dimensional input space

**Remark:** 

1. From a GP point of view, $k$ is the kernel of $Z(x) = Z(x_1) + Z(x_2) $.

2. It is straightforward to show that the mean predictor is additive.

$$
\begin{aligned} m(\mathbf{x}) &=\left(k_{1}(x, X)+k_{2}(x, X)\right)(k(X, X))^{-1} F \\ &=\underbrace{k_{1}\left(x_{1}, X_{1}\right)(k(X, X))^{-1} F}_{m_{1}\left(x_{1}\right)}+\underbrace{k_{2}\left(x_{2}, X_{2}\right)(k(X, X))^{-1} F}_{m_{2}\left(x_{2}\right)} \end{aligned}
$$

3. The prediction variance has interesting features.

   ![1560894134746](C:\Users\13269\Desktop\python_Markdown\GPSS\Notebook-for-GPSS2017\pics\02-11.PNG)

The right one comes from a additive kernel, as we can see even in the area which is away from the observation points, the variance is not too high. The reason for that our prior, e.g. kernel is additive, we already three observations which would form a rectangle, and our prediction would be the fourth vertex, thus the variance would be small.  **All the prior would retrieve it in the posterior**.

This property can be used to construct a design of experiment that covers the space  especially for the high-D input space,  with only $cst × d$ points.

![1560894678169](C:\Users\13269\Desktop\python_Markdown\GPSS\Notebook-for-GPSS2017\pics\02-12.PNG)

### Product  over  the  same  space

Property:
$$
k(x, y)=k_{1}(x, y) \times k_{2}(x, y)
$$
is valid covariance structure.

![1560894868103](C:\Users\13269\Desktop\python_Markdown\GPSS\Notebook-for-GPSS2017\pics\02-13.PNG)

### Product  over  the tensor  space

$$
k(\mathbf{x}, \mathbf{y})=k_{1}\left(x_{1}, y_{1}\right) \times k_{2}\left(x_{2}, y_{2}\right)
$$

![1560895062083](C:\Users\13269\Desktop\python_Markdown\GPSS\Notebook-for-GPSS2017\pics\02-14.PNG)

### Composition with  a  function 

$$
k(x, y)=k_{1}(f(x), f(y))\\
Proof:\\
\sum \sum a_{i} a_{j} k\left(x_{i}, x_{j}\right)=\sum \sum a_{i} a_{j} k_{1}\left(\underbrace{f\left(x_{i}\right)}_{y_{i}}, \underbrace{f\left(x_{j}\right)}_{y_{j}}\right) \geq 0
$$

This can be seen as a nonlinear rescaling of the input space.

## Periodicity  detection

Given a few observations can we extract the periodic part of a signal ?

As previously we will build a decomposition of the process in two independent GPs :
$$
Z=Z_{p}+Z_{a}\\
$$
${\text { where } Z_{p} \text { is a GP in the span of the Fourier basis }} $
$$
 {B(t)=(\sin (t), \cos (t), \ldots, \sin (n t), \cos (n t))^{t}}
$$
Note that the aperiodic means the projection of the $cos-sin$ space will end up with zero.

And it can be proved that
$$
\begin{array}{l}{k_{p}(x, y)=B(x)^{t} G^{-1} B(y)} \\ {k_{a}(x, y)=k(x, y)-k_{p}(x, y)}\end{array}
$$
where $G$ is the Gram Matrix associated to $B$ in the RKHS.

As previously, a decomposition of the model comes with a decomposition of the kernel:
$$
\begin{aligned} m(t)=&\left(k_{p}(x, X)+k_{a}(x, X)\right) k(X, X)^{-1} F \\=& k_{p}(x, X) k(X, X)^{-1} F+\underbrace{k_{a}(x, X) k(X, X)^{-1} F}_{\text { aperiodic sub-model } m_{a}} \end{aligned}
$$
and we can associate a prediction variance to the sub-models:
$$
\begin{aligned} v_{p}(t) &=k_{p}(x, x)-k_{p}(x, X)^{t} k(X, X)^{-1} k_{p}(t) \\ v_{a}(t) &=k_{a}(x, x)-k_{a}(x, X)^{t} k(X, X)^{-1} k_{a}(t) \end{aligned}
$$
![1560940503339](C:\Users\13269\Desktop\python_Markdown\GPSS\Notebook-for-GPSS2017\pics\02-15.PNG)