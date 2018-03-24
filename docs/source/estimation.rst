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
	\max_{\theta, \beta_1, ..., \beta_n} \sum_{i=1}^n \left \{ \log c(F_1(x_{i1}, \beta_1), ..., F_d(x_{id}, \beta_d)) + \sum_{j=1}^d \log f(x_{ij}, \beta_j) \right \}



For instance, suppose that we would like to fit a copula thanks to MLE method where :math:`X_1 \sim Gamma(\alpha, 1.2)` and :math:`X_2 \sim Gamma(1.8, \gamma)` with :math:`\alpha > 0` and :math:`\gamma > 0`. Then we would write the following code :

.. code-block:: python
   :emphasize-lines: 3,5

	mle(copula, X, marginals=[ scipy.stats.gamma, scipy.stats.gamma ], hyper_param=[ [None, 1.2], [1.8, None] ], hyper_param_bounds=[ [0, None], [0, None]])

Here is a detailled example on how to fit a Clayton copula with MLE.

.. code-block:: python
   :emphasize-lines: 3,5

	clayton = ArchimedeanCopula(family="clayton", dim=2)
	boundAlpha = [0, None] # Greater than 0
	boundGamma = [0, None]
	bounds = [ boundAlpha, boundGamma ]
	paramX1 = [None, 1.2] # Hyper-parameters of first Gamma
	paramX2 = [1.8, None] # Hyper-parameters of second Gamma
	hyperParams = [ paramX1, paramX2 ] # The hyper-parameters
	gamma = scipy.stats.gamma # The Gamma distribution
	# Fitting copula with MLE method and Gamma marginals distributions
	clayton.fit(data, method='mle', marginals=[gamma, gamma], hyper_param=hyperParams, hyper_param_bounds=bounds)

Keep in mind that, in case where there are many hyper-parameters, the computational cost can be extremely high.

Inference For Margins (IFM)
----------------------------

Canonical Maximum Likelihood Estimation (CMLE)
-----------------------------------------------
.. automodule:: estimation
	:members: cmle

