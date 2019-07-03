## GP-UKF: Unscented Kalman Filters with Gaussian Process Prediction and Observation Models
The training data for each GP is a set of input-output relations. The observation model maps from the state, $x_{k}$, to the observation, $z_{k}$. The appropriate form of the observation training data sets is:
$$D_{h} =  <X,Z>$$
where, $X$ is the matrix of ground truth matrix, $Z$ is the observed outputs from the camera.

One issue to consider when learning observation model is that GPs assume a zero mean prior. The idea is to use a GP to learn the **residual** between the true system model and the approximate parametric model. The combined parametric plus GP model is called an Enhanced-GP model.

Thus,
$$ z_{k} = \hat{h}({x_{k}}) + \widehat{GP_{\mu}}(x_{k},\widehat{D}_{k}) + \delta_{k}$$

with training data:

$$\widehat{D}_{h} = <X,Z-\widehat{h}(X)>$$

where, 

$h$: ground truth 

$\hat{h}$: approximate parametric model.

$\hat{h}$
