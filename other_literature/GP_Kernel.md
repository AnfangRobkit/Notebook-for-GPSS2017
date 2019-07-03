## Standard Kernels
### Squarded Expoential kernel
$$K = \sigma^{2} \exp (-\frac{(x-x^{'})}{2\mathit{l^{2}}}))$$
- The lengthscale ℓ determines the length of the 'wiggles' in your function. In general, you won't be able to extrapolate more than ℓ units away from your data.
- The output variance $σ^{2}$ determines the average distance of your function away from its mean. Every kernel has this parameter out in front; it's just a scale factor.

### Rational Quadratic Kernel
$$ k = \sigma^{2} (1 + \frac{r^{2}}{2\alpha\mathit{l}^{2}})^{-\alpha}$$
This kernel is equivalent to adding together many SE kernels with different lengthscales. So, GP priors with this kernel expect to see functions which vary smoothly across many lengthscales. The parameter α determines the relative weighting of large-scale and small-scale variations.

### Periodic Kernel
$$ K = \sigma^{2} \exp (-\frac{2 \sin^{2}(\pi r^{2}/p)}{2\mathit{l^{2}}})) $$

allows one to model functions which repeat themselves exactly. Its parameters are easily interpretable:
- The period p simply determines the distnace between repititions of the function.
- The lengthscale ℓ determines the lengthscale function in the same way as in the SE kernel.

### Locally Periodic Kernel
A SE kernel times a periodic results in functions which are periodic, but which can slowly vary over time.Most periodic functions don't repeat themselves exactly. To add some flexibility to our model, we can consider adding or multiplying a local kernel such as the squared-exp with our periodic kernel. This will allow us to model functions that are only locally periodic - the shape of the repeating part of the function can now change over time. 

## Combining Kernels
### Multiplying Kernels
Multiplying together kernels is the standard way to combine two kernels, especially if they are defined on different inputs to your function.
Roughly speaking, multiplying two kernels can be thought of as an AND operation. That is, if you multiply together two kernels, then the resulting kernel will have high value only if both of the two base kernels have a high value. 
#### Multidimensional Products
Multiplying two kernels which each depend only on a single input dimension results in a prior over functions that vary across both dimesions. That is, the function value f(x,y) is only expected to be similar to some other function value f(x′,y′) if x is close to $x^{′}$ **AND** y is close to $y^{′}$. 

These kernels have the form: 
$$K_{product}(x,y,x^{′},y^{′})=K_{x}(x,x^{})k_{}(y,y^{'})$$ 

### Adding Kernels
Roughly speaking, adding two kernels can be thought of as an OR operation. That is, if you add together two kernels, then the resulting kernel will have high value if either of the two base kernels have a high value.

## Other Types of Structure
### Encode Symmetry into the kernel
In some cases, we know that the function we're modeling is symmetric. There turns out to be an simple trick to ensure that all the functions you consider are symmetric: simply add the kernel to itself, but with the order of the inputs swapped.
#### Axis-aligned reflective symmetry
For example, to enforce that $f(x)=f(−x)$, simply transform your kernel like so: $k_{1d_Symmetry}(x,x^{′})=k(x,x^{′})+k(−x,x^{'})$ 
#### Enforcing independence from the order of arguments
To enforce that $f(x,y)=f(y,x)$, use this kernel transformation: $k_{2d_Symmetry}(x,y,x^{′},y^{′})=k(x,y,x^{′},y^{′})+k(y,x,x^{′},y^{'})$ 
## References
https://www.cs.toronto.edu/~duvenaud/cookbook/