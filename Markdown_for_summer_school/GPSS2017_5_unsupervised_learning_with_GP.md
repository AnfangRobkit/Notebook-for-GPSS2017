$$
\begin{array}{c}{p(y)=\int p(y | f) p(f | x) p(x) \mathrm{d} f \mathrm{d} x} \\ {p(x | y)=p(y | x) \frac{p(x)}{p(y)}}\end{array}
$$

1. Priors that makes sense:

   - `p(f)` describes our belief/assumptions and defines our notion of complexity in the function
   - p(x) expresses our belief/assumptions and defines our notion of complexity in the latent space

2. The priors are balanced.

   

   GP prior:
   $$
   \begin{aligned} p(f | x) & \sim \mathcal{N}(0, K) \propto e^{-\frac{1}{2}\left(f^{\mathrm{T}} K^{-1} f\right)} \\ K_{i j} &=e^{-\left(x_{i}-x_{j}\right)^{\mathrm{T}} M^{\mathrm{T}} M\left(x_{i}-x_{j}\right)} \end{aligned}
   $$
   Likelihood:

$$
p(y | f) \sim N(y | f, \beta) \propto e^{-\frac{1}{2 \beta} \operatorname{tr}(y-f)^{\mathrm{T}}(y-f)}
$$

Analytically intractable (==Non Elementary Integral==) and infinitely differentiable. One way to avoid the Integral is to use:
$$
\begin{array}{c}{\hat{x}=\operatorname{argmax}_{x} \int p(y | f) p(f | x) \mathrm{d} f p(x)} \\ {=\operatorname{argmin}_{x} \frac{1}{2} y^{\mathrm{T}} \mathbf{K}^{-1} y+\frac{1}{2}|\mathbf{K}|-\log p(x)}\end{array}
$$

Challenges with ML estimation:

- how to initialize `x`?
- What is the dimensionality `q`? which means how complex the latent space should be to represent `y`?

## Variational Bayes:

$$
\begin{aligned} \log p(\mathbf{Y}) &=\log \int p(\mathbf{Y}, \mathbf{X}) \mathrm{d} \mathbf{X}=\log \int p(\mathbf{X} | \mathbf{Y}) p(\mathbf{Y}) \mathrm{d} \mathbf{X} \\ &=\log \int \frac{q(\mathbf{X})}{q(\mathbf{X})} p(\mathbf{X} | \mathbf{Y}) p(\mathbf{Y}) \mathrm{d} \mathbf{X} \end{aligned}
$$

For a convex function:
$$
\begin{aligned} \lambda f\left(x_{0}\right)+(1-\lambda) f\left(x_{1}\right) & \geq f\left(\lambda x_{0}+(1-\lambda) x_{1}\right) \\ x & \in\left[x_{\min }, x_{\max }\right] \\ \lambda & \in[0,1] ] \end{aligned}
$$
In probability, that means:
$$
\begin{aligned} \mathbb{E}[f(x)] & \geq f(\mathbb{E}[x]) \\ \int f(x) p(x) \mathrm{d} x & \geq f\left(\int x p(x) \mathrm{d} x\right) \end{aligned}\\
\int \log (x) p(x) \mathrm{d} x \leq \log \left(\int x p(x) \mathrm{d} x\right)
$$
thus, 
$$
\begin{aligned} \log p(\mathbf{Y}) &=\log \int \frac{q(\mathbf{X})}{q(\mathbf{X})} p(\mathbf{X} | \mathbf{Y}) p(\mathbf{Y}) \mathrm{d} \mathbf{X}=\\ & \geq \int q(\mathbf{X}) \log \frac{p(\mathbf{X} | \mathbf{Y}) p(\mathbf{Y})}{q(\mathbf{X})} \mathrm{d} \mathbf{X} \\ &=\int q(\mathbf{X}) \log \frac{p(\mathbf{X} | \mathbf{Y})}{q(\mathbf{X})} \mathrm{d} \mathbf{X}+\int q(\mathbf{X}) \mathrm{d} \mathbf{X} \log p(\mathbf{Y}) \\ &=-\mathrm{KL}(q(\mathbf{X}) \| p(\mathbf{X} | \mathbf{Y}))+\log p(\mathbf{Y}) \end{aligned}
$$
$KL$ is KL-Divergence that is a measure of how one probability distribution is different from a second, reference probability distribution.

If $q(x)$ is the true posterior we have an equality, therefore match the distributions.
$$
\begin{aligned} \mathrm{KL}(q(\mathbf{X}) \| p(\mathbf{X} | \mathbf{Y})) &=\int q(\mathbf{X}) \log \frac{q(\mathbf{X})}{p(\mathbf{X} | \mathbf{Y})} \mathrm{d} \mathbf{X} \\ &=\int q(\mathbf{X}) \log \frac{q(\mathbf{X})}{p(\mathbf{X}, \mathbf{Y})} \mathrm{d} \mathbf{X}+\log p(\mathbf{Y}) \\ &=H(q(\mathbf{X}))-\mathbb{E}_{q(\mathbf{X})}[\log p(\mathbf{X}, \mathbf{Y})]+\log p(\mathbf{Y}) \end{aligned}
$$
And we rearrange it:
$$
\begin{aligned} \log p(\mathbf{Y}) &=\mathrm{KL}(q(\mathbf{X}) \| p(\mathbf{X} | \mathbf{Y}))+\underbrace{\mathbb{E}_{q(\mathbf{X})}[\log p(\mathbf{X}, \mathbf{Y})]-H(q(\mathbf{X}))}_{\text { ELBO }} \\ & \geq \mathbb{E}_{q(\mathbf{X})}[\log p(\mathbf{X}, \mathbf{Y})]-H(q(\mathbf{X}))=\mathcal{L}(q(\mathbf{X})) \end{aligned}
$$
if we maximize the ELBO, it means:

- find an approximate posterior
- get an approximate to the marginal likelihood

**Maximizing $p(Y)$ is learning**

finding $p(X|Y) \sim p(X)$ is **prediction**

