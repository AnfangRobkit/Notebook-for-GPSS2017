## GP Parameter

- Amplitude ($\sigma$) controls the scaling of the output along the y-axis. This parameter is just a scalar multiplier, and is therefore usually left out of implementations of the Matèrn function (i.e. set to one). Amplitude is an included parameter (variance), so we do not need to include a separate constant kernel.
  
- lengthscale ($l$) complements the amplitude by scaling realizations on the x-axis. Larger values push points closer together along this axis.

- roughness ($\nu$) controls the sharpness of ridges in the covariance function, which ultimately affect the roughness (smoothness) of realizations. In $GPy$, there are two subclasses: one which fixes the roughness parameter to 3/2 ($Matern32$) and another to 5/2 ($Matern52$). 