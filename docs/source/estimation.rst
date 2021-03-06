Estimation
==========
Our goal is to estimate the parameter :math:`\theta` of a copula, where :math:`X_1, ..., X_d` are random variables associated to the marginals. We denote :math:`F_1, ..., F_d` the cumulative distribution functions (CDF) and :math:`f_1, ..., f_d` the probability density functions (PDF), if it exists, of those random variables. Most of estimation methods involve likelihood function :math:`L`, that we can easily get using the expression of copula's density :math:`c` :

.. math::
	L(x_1, ..., x_d) = c(F_1(x_1), ..., F_d(x_d)) \prod_{i=1}^d f_i(x_i)

To fit the copulas, we use a dataset :math:`\mathbf{x}=\left \{(x_{i1}, ..., x_{id}) \right \}_{1 \leq i \leq n}` and the log-likelihood function will refer to :math:`\mathcal{L}(\mathbf{x}_i)=\log L(\mathbf{x}_i)`.

Maximum Likelihood Estimation (MLE)
------------------------------------

.. automodule:: estimation
	:members: mle

The MLE objective is to maximize the log-likelihood function over all parameters and hyper-parameters of marginals. We suppose that :math:`X_j \sim f(\beta_j)` where :math:`\beta_j` is an hyper-parameter of the copula. The MLE will then return the copula's parameter and all estimated hyper-parameters at the same time. The estimated parameters :math:`\hat{\theta}, \hat{\beta}_1, ..., \hat{\beta}_d` are solution to the following optimization problem.

.. math::
	\max_{\theta, \beta_1, ..., \beta_d} \sum_{i=1}^n \left \{ \log c(F_1(x_{i1}, \beta_1), ..., F_d(x_{id}, \beta_d)) + \sum_{j=1}^d \log f(x_{ij}, \beta_j) \right \}



For instance, suppose that we would like to fit a copula thanks to MLE method where :math:`X_1 \sim Gamma(\alpha, 1.2)` and :math:`X_2 \sim Exp(\lambda)` with :math:`\alpha > 0` and :math:`\lambda > 0`. Then we would write the following code :

.. code-block:: pythons
   :emphasize-lines: 3,5

	mle(copula, X, marginals=[ scipy.stats.gamma, scipy.stats.expon ], hyper_param=[ { 'a': None, 'scale': 1.2 }, { 'scale': None } ], hyper_param_bounds=[ [0, None], [0, None]])

Use None to consider an hyper-parameter as unknown and None to define :math:`\pm \infty` in hyper-parameters bounds. Here is a detailled example on how to fit a Clayton copula with MLE.

.. code-block:: python
   :emphasize-lines: 3,5

	clayton = ArchimedeanCopula(family="clayton", dim=2)
	boundAlpha = [0, None] # Greater than 0
	boundLambda = [0, None]
	bounds = [ boundAlpha, boundLambda ]
	paramX1 = { 'a': None, 'scale': 1.2 } # Hyper-parameters of Gamma
	paramX2 = { 'scale': None } # Hyper-parameters of Exp
	hyperParams = [ paramX1, paramX2 ] # The hyper-parameters
	gamma = scipy.stats.gamma # The Gamma distribution
	expon = scipy.stats.expon # The Exponential distribution
	# Fitting copula with MLE method and Gamma/Exp marginals distributions
	clayton.fit(data, method='mle', marginals=[gamma, exp], hyper_param=hyperParams, hyper_param_bounds=bounds)

Keep in mind that, in case where there are many hyper-parameters, the computational cost can be extremely high.

Inference Functions for Margins (IFM)
--------------------------------------
.. automodule:: estimation
	:members: ifm

The difference with the previous method is that hyper-parameters are estimated independently from the copula's parameter.

.. math::
	\forall 1 \leq j \leq d, \hat{\beta}_j = \text{argmax}_{\beta_j} \sum_{i=1}^n \log f_i(x_{ij}, \beta_j)

Then, our observations :math:`\mathbf{x}` are transformed into uniform variables :math:`\mathbf{u}`.

.. math::
	\forall 1 \leq i \leq n, \forall 1 \leq i \leq d, u_{ij}=F_j(x_{ij}, \hat{\beta_j})

Finally, as we did before, we compute the likelihood function and use optimization algorithm to estimate the parameter :math:`\theta`.

.. math::
	\hat{\theta} = \text{argmax}_{\theta} \sum_{i=1}^n \log c(u_{i1}, ..., u_{id}, \theta)

The specifications of this method are the same of MLE, and you can use this method calling fitting process.

.. code-block:: python
   :emphasize-lines: 3,5

	clayton.fit(data, method='ifm', marginals=[gamma, exp], hyper_param=hyperParams, hyper_param_bounds=bounds)


Canonical Maximum Likelihood Estimation (CMLE)
-----------------------------------------------
.. automodule:: estimation
	:members: cmle

This semi-parametric method does not require to specify the marginals distributions of the copula. Indeed, instead of estimating the hyper-parameters, the empirical CDF :math:`\hat{F}_j` for each random variable is computed and observations :math:`\mathbf{x}` are transformed into uniform variables :math:`\mathbf{u}`.

.. math::
	\forall 1 \leq i \leq n, \forall 1 \leq j \leq d, u_{ij}=\hat{F}_j(x_{ij})

Estimating the parameter of the copula is then applied maximizing log-likelihood on transformed data.

.. math::
	\hat{\theta} = \text{argmax}_{\theta} \sum_{i=1}^n \log c(u_{i1}, ..., u_{id}, \theta)

Since CMLE does not require marginal distributions, using this method is quite easy. For instance, on the Clayton copula :

.. code-block:: python
   :emphasize-lines: 3,5

	clayton.fit(data, method='cmle')

