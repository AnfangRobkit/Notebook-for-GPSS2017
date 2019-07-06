In GPy we deal with multiple output data in a particular way. We specify the output we are interested in for modelling as an additional input.

The linear models of coregionalization we introduced in the lecture were all based on combining a matrix with a standard covariance function. We can think of the matrix as a particular type of covariance function, whose elements are referenced using the event indices. I.e. $k(0,0)$ references the first row and column of the coregionalization matrix. $k(1,0)$ references the second row and first column of the coregionalization matrix. Under this set up, we want to build a covariance where the first column from the features (the years) is passed to a covariance function, and the second column from the features (the event number) is passed to the coregionalisation matrix ${\bf B}$. 

 ${\bf B} = {\bf W}{\bf W}^\top + {\boldsymbol \kappa}{\bf I}$