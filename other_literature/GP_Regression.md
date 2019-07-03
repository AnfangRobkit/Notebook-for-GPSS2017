## Predictions from posterior

#### https://peterroelants.github.io/posts/gaussian-process-tutorial/

We want to make predictions $y_{2} = f(X_{2})$ for $n_{2}$ new samples, and we want to make these predictions based on our Gaussian process prior and $n_{1}$ previously observed data points . This can be done with the help of the posterior distribution $p(y_{2}|y_{1},X_{1},X_{2})$. Keep in mind that $y_{1} y_{2}$
 are jointly Gaussian since they both should come from the same function.
 
 Then we can get the conditional distribution:
 $p(y_{2}|y_{1},X_{1},X_{2}) = \mathcal{N}(\mu_{2|1},\varSigma_{2|1})$
 $\mu_{2|1} = \mu_{2}+\varSigma_{21}\varSigma_{11}^{-1}(y_{1}-\mu_{1})$
 $\mu_{2|1}=\varSigma_{2|1}\varSigma_
 {11}^{-1}y_{1}$
 $\varSigma_{2|1}=\varSigma_{22}-\varSigma_{21}\varSigma_{11}^{-1}\varSigma_{12}$
 if we assume mean piror $\mu = 0$

 We can write these as follows：
 $μ_{2|1}=Σ_{21}Σ^{−1}_{11}y_{1}=(Σ_{11}^{−1}Σ12)^{⊤}y1$

## Noisy observations
We can make predictions from noisy observations $f(X_{1})=Y_{1}+\epsilon$, by modelling the noise as Gaussian noise with variance $\sigma_{\epsilon}^{2}$.

This noise can be modelled by adding it to the covariance kernel:
$$\varSigma_{11}=k(X_{1},X_{1})+\sigma_{\epsilon}^{2}\mathcal{I}$$

Note that the noise only changes kernel values on the diagonal (white noise is independently distributed). 