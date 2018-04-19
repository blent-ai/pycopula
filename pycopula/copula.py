#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	This file contains all the classes for copula objects.
"""

__author__ = "Maxime Jumelle"
__copyright__ = "Copyright 2018, AIPCloud"
__credits__ = "Maxime Jumelle"
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Maxime Jumelle"
__email__ = "maxime@aipcloud.io"
__status__ = "Development"

import sys

sys.path.insert(0, '..')

import archimedean_generators as generators
import math_misc
from math_misc import multivariate_t_distribution
import estimation

import scipy
from scipy.stats import kendalltau, pearsonr, spearmanr, norm, multivariate_normal
from scipy.linalg import sqrtm
import scipy.misc

import numpy as np
from numpy.linalg import inv

# An abstract copula object
class Copula():

	def __init__(self, dim=2, name='indep'):
		"""
		Creates a new abstract Copula.
		
		Parameters
		----------
		dim : integer (greater than 1)
			The dimension of the copula.
		name : string
			Default copula. 'indep' is for independency copula, 'frechet_up' the upper Fréchet-Hoeffding bound and 'frechet_down' the lower Fréchet-Hoeffding bound.
		"""
		if dim < 2 or int(dim) != dim:
			raise ValueError("Copula dimension must be an integer greater than 1.")
		self.dim = dim
		self.name = name
		self.kendall = None
		self.pearson = None
		self.spearman = None

	def __str__(self):
		return "Copula ({0}).".format(self.name)

	def _checkDimension(self, x):
		if len(x) != self.dim:
			raise ValueError("Expected vector of dimension {0}, get vector of dimension {1}".format(self.dim, len(x)))
	
	def getDimension(self):
		"""
		Returns the dimension of the copula.
		"""
		return self.dim

	def correlations(self, X):
		"""
		Compute the correlations of the specified data. Only available when dimension of copula is 2.

		Parameters
		----------
		X : numpy array (of size n * 2)
			Values to compute correlations.

		Returns
		-------
		kendall : float
			The Kendall tau.
		pearson : float
			The Pearson's R
		spearman : float
			The Spearman's R
		"""
		if self.dim != 2:
			raise Exception("Correlations can not be computed when dimension is greater than 2.")
		self.kendall = kendalltau(X[:,0], X[:,1])[0]
		self.pearson = pearsonr(X[:,0], X[:,1])[0]
		self.spearman = spearmanr(X[:,0], X[:,1])[0]
		return self.kendall, self.pearson, self.spearman

	def kendall(self):
		"""
		Returns the Kendall's tau. Note that you should previously have computed correlations.
		"""
		if self.kendall == None:
			raise ValueError("You must compute correlations before accessing to Kendall's tau.")
		return self.kendall

	def pearson(self):
		"""
		Returns the Pearson's r. Note that you should previously have computed correlations.
		"""
		if self.pearson == None:
			raise ValueError("You must compute correlations before accessing to Pearson's r.")
		return self.pearson

	def spearman(self):
		"""
		Returns the Spearman's rho. Note that you should previously have computed correlations.
		"""
		if self.pearson == None:
			raise ValueError("You must compute correlations before accessing to Spearman's rho.")
		return self.spearman	

	def cdf(self, x):
		"""
		Returns the cumulative distribution function (CDF) of the copula.
		
		Parameters
		----------
		x : numpy array (of size d)
			Values to compute CDF.
		"""
		self._checkDimension(x)
		if self.name == 'indep':
			return np.prod(x)
		elif self.name == 'frechet_up':
			return min(x)
		elif self.name == 'frechet_down':
			return max(sum(x) - self.dim + 1., 0)

	def pdf(self, x):
		"""
		Returns the probability distribution function (PDF) of the copula.
		
		Parameters
		----------
		x : numpy array (of size d)
			Values to compute PDF.
		"""
		self._checkDimension(x)
		if self.name == 'indep':
			return sum([ np.prod([ x[j] for j in range(self.dim) if j != i ]) for i in range(self.dim) ])
		elif self.name in [ 'frechet_down', 'frechet_up' ]:
			raise NotImplementedError("PDF is not available for Fréchet-Hoeffding bounds.")
			
	def concentrationDown(self, x):
		"""
		Returns the theoritical lower concentration function.
		
		Parameters
		----------
		x : float (between 0 and 0.5)
		"""
		if x > 0.5 or x < 0:
			raise ValueError("The argument must be included between 0 and 0.5.")
		return self.cdf([x, x]) / x
		
	def concentrationUp(self, x):
		"""
		Returns the theoritical upper concentration function.
		
		Parameters
		----------
		x : float (between 0.5 and 1)
		"""
		if x < 0.5 or x > 1:
			raise ValueError("The argument must be included between 0.5 and 1.")
		return (1. - 2*x + self.cdf([x, x])) / (1. - x)
		
	def concentrationFunction(self, x):
		"""
		Returns the theoritical concentration function.
		
		Parameters
		----------
		x : float (between 0 and 1)
		"""
		if x < 0 or x > 1:
			raise ValueError("The argument must be included between 0 and 1.")
		if x < 0.5:
			return self.concentrationDown(x)
		return self.concentrationUp(x)
		
class ArchimedeanCopula(Copula):

	families = [ 'clayton', 'gumbel', 'frank', 'joe', 'amh' ]

	def __init__(self, family='clayton', dim=2):
		"""
		Creates an Archimedean copula.
		
		Parameters
		----------
		family : str
			The name of the copula.
		dim : int
			The dimension of the copula.
		"""
		super(ArchimedeanCopula, self).__init__(dim=dim)
		self.family = family
		self.parameter = 2.0
		if family == 'clayton':
			self.generator = generators.claytonGenerator
			self.generatorInvert = generators.claytonGeneratorInvert
		elif family == 'gumbel':
			self.generator = generators.gumbelGenerator
			self.generatorInvert = generators.gumbelGeneratorInvert
		elif family == 'frank':
			self.generator = generators.frankGenerator
			self.generatorInvert = generators.frankGeneratorInvert
		elif family == 'joe':
			self.generator = generators.joeGenerator
			self.generatorInvert = generators.joeGeneratorInvert
		elif family == 'amh':
			self.parameter = 0.5
			self.generator = generators.aliMikhailHaqGenerator
			self.generatorInvert = generators.aliMikhailHaqGeneratorInvert
		else:
			raise ValueError("The family name '{0}' is not defined.".format(family))

	def __str__(self):
		return "Archimedean Copula ({0}) :".format(self.family) + "\n*\tParameter : {:1.6f}".format(self.parameter)
		
	def generator(self, x):
		return self.generator(x, self.parameter)
		
	def inverseGenerator(self, x):
		return self.generatorInvert(x, self.parameter)
		
	def getParameter(self):
		return self.parameter
		
	def setParameter(self, theta):
		self.parameter = theta
		
	def getFamily(self):
		return self.family

	def _checkDimension(self, x):
		"""
		Check if the number of variables is equal to the dimension of the copula.
		"""
		if len(x) != self.dim:
			raise ValueError("Expected vector of dimension {0}, get vector of dimension {1}".format(self.dim, len(x)))

	def cdf(self, x):
		"""
		Returns the CDF of the copula.

		Parameters
		----------
		x : numpy array (of size copula dimension or n * copula dimension)
			Quantiles.

		Returns
		-------
		float
			The CDF value on x.
		"""
		if len(np.asarray(x).shape) > 1:
			self._checkDimension(x[0])
			return [ self.generatorInvert(sum([ self.generator(v, self.parameter) for v in row ]), self.parameter) for row in x ]
		else:
			self._checkDimension(x)
			return self.generatorInvert(sum([ self.generator(v, self.parameter) for v in x ]), self.parameter)

	def pdf_param(self, x, theta):
		"""
		Returns the PDF of the copula with the specified theta. Use this when you want to compute PDF with another parameter.

		Parameters
		----------
		x : numpy array (of size n * copula dimension)
			Quantiles.
		theta : float
			The custom parameter.

		Returns
		-------
		float
			The PDF value on x.
		"""
		self._checkDimension(x)
		# prod is the product of the derivatives of the generator for each variable
		prod = 1
		# The sum of generators that will be computed on the invert derivative
		sumInvert = 0
		# The future function (if it exists) corresponding to the n-th derivative of the invert
		invertNDerivative = None

		# For each family, the structure is the same
		if self.family == 'clayton':
			# We compute product and sum
			for i in range(self.dim):
				prod *= -x[i]**(-theta - 1.)
				sumInvert += self.generator(x[i], theta)

			# We define (when possible) the n-th derivative of the invert of the generator
			def claytonInvertnDerivative(t, theta, order):
				product = 1
				for i in range(1, order):
					product *= (-1. / theta - i)
				return -theta**(order - 1) * product * (1. + theta * t)**(-1. / theta - order)

			invertNDerivative = claytonInvertnDerivative	

		elif self.family == 'gumbel':
			if self.dim == 2:
				for i in range(self.dim):
					prod *= (theta / (np.log(x[i]) * x[i]))*(-np.log(x[i]))**theta
					sumInvert += self.generator(x[i], theta)

				def gumbelInvertDerivative(t, theta, order):
					return 1. / theta**2 * t**(1. / theta - 2.) * (theta + t**(1. / theta) - 1.) * np.exp(-t**(1. / theta))
				
				if self.dim == 2:
					invertNDerivative = gumbelInvertDerivative
				
				
		elif self.family == 'frank':
			if self.dim == 2:
				for i in range(self.dim):
					prod *= theta / (1. - np.exp(theta * x[i]))
					sumInvert += self.generator(x[i], theta)

				def frankInvertDerivative(t, theta, order):
					C = np.exp(-theta) - 1.
					return - C / theta * np.exp(t) / (C + np.exp(t))**2
					
				invertNDerivative = frankInvertDerivative
				
		elif self.family == 'joe':
			if self.dim == 2:
				for i in range(self.dim):
					prod *= -theta * (1. - x[i])**(theta - 1.) / (1. - (1. - x[i])**theta)
					sumInvert += self.generator(x[i], theta)

				def joeInvertDerivative(t, theta, order):
					return 1. / theta**2 * (1. - np.exp(-t))**(1. / theta) * (theta * np.exp(t) - 1.) / (np.exp(t) - 1.)**2

				invertNDerivative = joeInvertDerivative
				
		elif self.family == 'amh':
			if self.dim == 2:
				for i in range(self.dim):
					prod *= (theta - 1.) / (x[i] * (1. - theta * (1. - x[i])))
					sumInvert += self.generator(x[i], theta)

				def amhInvertDerivative(t, theta, order):
					return (1. - theta) * np.exp(t) * (theta + np.exp(t)) / (np.exp(t) - theta)**3

				invertNDerivative = amhInvertDerivative
				
		if invertNDerivative == None:
			try:
				invertNDerivative = lambda t, theta, order: scipy.misc.derivative(lambda x: self.generatorInvert(x, theta), t, n=order, order=order+order%2+1)
			except:
				raise Exception("The {0}-th derivative of the invert of the generator could not be computed.".format(self.dim))
		
		# We compute the PDF of the copula
		return prod * invertNDerivative(sumInvert, theta, self.dim)

	def pdf(self, x):
		return self.pdf_param(x, self.parameter)

	def fit(self, X, method='cmle', verbose=False, theta_bounds=None, **kwargs):
		"""
		Fit the archimedean copula with specified data.

		Parameters
		----------
		X : numpy array (of size n * copula dimension)
			The data to fit.
		method : str
			The estimation method to use. Default is 'cmle'.
		verbose : bool
			Output various informations during fitting process.
		theta_bounds : tuple
			Definition set of theta. Use this only with custom family.
		**kwargs
			Arguments of method. See estimation for more details.

		Returns
		-------
		float
			The estimated parameter of the archimedean copula.
		estimationData
			Various data from estimation method. Often estimated hyper-parameters.
		
		"""
		n = X.shape[0]
		if n < 1:
			raise ValueError("At least two values are needed to fit the copula.")
		self._checkDimension(X[0,:])
		estimationData = None
		
		# Moments method (only when dimension = 2)
		if method == 'moments':
			if self.kendall == None:
				self.correlations(X)
			if self.family == 'clayton':
				self.parameter = 2. * self.kendall / (1. - self.kendall)
			elif self.family == 'gumbel':
				self.parameter = 1. / (1. - self.kendall)
			else:
				raise Exception("Moments estimation is not available for this copula.")
			
			
		# Canonical Maximum Likelihood Estimation
		elif method == 'cmle':
			# Pseudo-observations from real data X
			pobs = []
			for i in range(self.dim):
				order = X[:,i].argsort()
				ranks = order.argsort()
				u_i = [ (r + 1) / (n + 1) for r in ranks ]
				pobs.append(u_i)

			pobs = np.transpose(np.asarray(pobs))

			bounds = theta_bounds
			if bounds == None:
				if self.family == 'amh':
					bounds = (-1, 1 - 1e-6)
				elif self.family == 'clayton':
					bounds = (-1, None)
				elif self.family in ['gumbel', 'joe'] :

					bounds = (1, None)

			def log_likelihood(theta):
				return sum([ np.log(self.pdf_param(pobs[i,:], theta)) for i in range(n) ])

			theta_start = np.array(2.)
			if self.family == 'amh':
				theta_start = np.array(0.5)

			res = estimation.cmle(log_likelihood, theta_start=theta_start, theta_bounds=bounds, optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'), bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
			self.parameter = res['x'][0]
		
		# Maximum Likelihood Estimation and Inference Functions for Margins
		elif method in [ 'mle', 'ifm' ]:
			if not('marginals' in kwargs):
				raise ValueError("Marginals distribution are required for MLE.")
			if not('hyper_param' in kwargs):
				raise ValueError("Hyper-parameters are required for MLE.")
			
			bounds = theta_bounds
			if bounds == None:
				if self.family == 'amh':
					bounds = (-1, 1 - 1e-6)
				elif self.family == 'clayton':
					bounds = (-1, None)
				elif self.family in ['gumbel', 'joe'] :
					bounds = (1, None)

			theta_start = [ 2 ]
			if self.family == 'amh':
				theta_start = [ 0.5 ]
				
			if method == 'mle':
				res, estimationData = estimation.mle(self, X, marginals=kwargs.get('marginals', None), hyper_param=kwargs.get('hyper_param', None), hyper_param_start=kwargs.get('hyper_param_start', None), hyper_param_bounds=kwargs.get('hyper_param_bounds', None), theta_start=theta_start, theta_bounds=bounds, optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'), bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
			else:
				res, estimationData = estimation.ifm(self, X, marginals=kwargs.get('marginals', None), hyper_param=kwargs.get('hyper_param', None), hyper_param_start=kwargs.get('hyper_param_start', None), hyper_param_bounds=kwargs.get('hyper_param_bounds', None), theta_start=theta_start, theta_bounds=bounds, optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'), bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
				
			self.parameter = res['x'][0]
		else:
			raise ValueError("Method '{0}' is not defined.".format(method))
			
		return self.parameter, estimationData
		
class GaussianCopula(Copula):
	def __init__(self, dim=2, sigma=[[1, 0.8], [0.8, 1]]):
		super(GaussianCopula, self).__init__(dim=dim)
		self.setCovariance(sigma)

	def cdf(self, x):
		self._checkDimension(x)
		return multivariate_normal.cdf([ norm.ppf(u) for u in x ], cov=self.sigma)

	def setCovariance(self, sigma):
		"""
		Set the covariance of the copula.

		Parameters
		----------
		sigma : numpy array (of size copula dimensions * copula dimension)
			The definite positive covariance matrix. Note that you should check yourself if the matrix is definite positive.
		"""
		S = np.asarray(sigma)
		if len(S.shape) > 2:
			raise ValueError("2-dimensional array expected, get {0}-dimensional array.".format(len(S.shape)))
		if S.shape[0] != S.shape[1]:
			raise ValueError("Covariance matrix must be a squared matrix of dimension {0}".format(self.dim))
		if len([ 1 for i in range(S.shape[0]) if S[i, i] <= 0]) > 0:
			raise ValueError("Null or negative variance encountered in covariance matrix.")
		if not(np.array_equal(np.transpose(S), S)):
			raise ValueError("Covariance matrix is not symmetric.")
		self.sigma = S
		self.sigmaDet = np.linalg.det(S)
		self.sigmaInv = np.linalg.inv(S)
		
	def getCovariance(self):
		return self.sigma

	def pdf(self, x):
		self._checkDimension(x)
		#return self.pdf_param(x, self.sigma)
		u_i = norm.ppf(x)
		return self.sigmaDet**(-0.5) * np.exp(-0.5 * np.dot(u_i, np.dot(self.sigmaInv - np.identity(self.dim), u_i)))
		
	def pdf_param(self, x, sigma):
		self._checkDimension(x)
		if self.dim == 2 and not(hasattr(sigma, '__len__')):
			sigma = [sigma]
		if len(np.asarray(sigma).shape) == 2 and len(sigma) != self.dim:
			raise ValueError("Expected covariance matrix of dimension {0}.".format(self.dim))
		u = norm.ppf(x)
		
		cov = np.ones([ self.dim, self.dim ])
		idx = 0
		if len(np.asarray(sigma).shape) <= 1:
			if len(sigma) == self.dim * (self.dim - 1) / 2:
				for j in range(self.dim):
					for i in range(j + 1, self.dim):
						cov[j][i] = sigma[idx]
						cov[i][j] = sigma[idx]
						idx += 1
			else:
				raise ValueError("Expected covariance matrix, get an array.")
		
		if self.dim == 2:
			sigmaDet = cov[0][0] * cov[1][1] - cov[0][1]**2
			sigmaInv = 1. / sigmaDet * np.asarray([[ cov[1][1], -cov[0][1]], [ -cov[0][1], cov[0][0] ]])
		else:
			sigmaDet = np.linalg.det(cov)
			sigmaInv = np.linalg.inv(cov)
		return [ sigmaDet**(-0.5) * np.exp(-0.5 * np.dot(u_i, np.dot(sigmaInv - np.identity(self.dim), u_i))) for u_i in u ]
		
	def quantile(self,  x):
		return multivariate_normal.ppf([ norm.ppf(u) for u in x ], cov=self.sigma)

	def fit(self, X, method='cmle', verbose=True, **kwargs):
		"""
		Fit the Gaussian copula with specified data.

		Parameters
		----------
		X : numpy array (of size n * copula dimension)
			The data to fit.
		method : str
			The estimation method to use. Default is 'cmle'.
		verbose : bool
			Output various informations during fitting process.
		**kwargs
			Arguments of method. See estimation for more details.

		Returns
		-------
		float
			The estimated parameters of the Gaussian copula.
		"""
		print("Fitting Gaussian copula.")
		n = X.shape[0]
		if n < 1:
			raise ValueError("At least two values are needed to fit the copula.")
		self._checkDimension(X[0, :])

		# Canonical Maximum Likelihood Estimation
		if method == 'cmle':
			# Pseudo-observations from real data X
			pobs = []
			for i in range(self.dim):
				order = X[:,i].argsort()
				ranks = order.argsort()
				u_i = [ (r + 1) / (n + 1) for r in ranks ]
				pobs.append(u_i)

			pobs = np.transpose(np.asarray(pobs))
			# The inverse CDF of the normal distribution (do not place it in loop, hungry process)
			ICDF = norm.ppf(pobs)

			def log_likelihood(rho):
				S = np.identity(self.dim)
				
				# We place rho values in the up and down triangular part of the covariance matrix
				for i in range(self.dim - 1):
					for j in range(i + 1,  self.dim):
						S[i][j] = rho[i * (self.dim - 1) + j - 1]
						S[self.dim - i - 1][self.dim - j - 1] = S[i][j]
				
				# Computation of det and invert matrix
				if self.dim == 2:
					sigmaDet = S[0][0] * S[1][1] - rho**2
					sigmaInv = 1. / sigmaDet * np.asarray([[ S[1][1], -rho], [ -rho, S[0][0] ]])
				else:
					sigmaDet = np.linalg.det(S)
					sigmaInv = np.linalg.inv(S)
				
				# Log-likelihood
				lh = 0
				for i in range(n):
					cDens = sigmaDet**(-0.5) * np.exp(-0.5 * np.dot(ICDF[i,  :], np.dot(sigmaInv - np.identity(self.dim), ICDF[i,  :])))
					lh += np.log(cDens)

				return lh

			rho_start = [ 0.0 for i in range(int(self.dim * (self.dim - 1) / 2)) ]
			res = estimation.cmle(log_likelihood, theta_start=rho_start, theta_bounds=None, optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'), bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
			rho = res['x']
		elif method == 'mle':
			rho_start = [ 0.0 for i in range(int(self.dim * (self.dim - 1) / 2)) ]
			res, estimationData = estimation.mle(self, X, marginals=kwargs.get('marginals', None), hyper_param=kwargs.get('hyper_param', None), hyper_param_start=kwargs.get('hyper_param_start', None), hyper_param_bounds=kwargs.get('hyper_param_bounds', None), theta_start=rho_start, optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'), bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
			rho = res['x']
			
		self.sigma = np.identity(self.dim)
		# We extract rho values to covariance matrix
		for i in range(self.dim - 1):
			for j in range(i + 1,  self.dim):
				self.sigma[i][j] = rho[i * (self.dim - 1) + j - 1]
				self.sigma[self.dim - i - 1][self.dim - j - 1] = self.sigma[i][j]
			
		# We compute the nearest semi-definite positive matrix for the covariance matrix
		self.sigma = math_misc.nearPD(self.sigma)
		self.setCovariance(self.sigma)

class StudentCopula(Copula):
	
	def __init__(self, dim=2, df=1, sigma=[[1, 0.6], [0.6, 1]]):
		super(StudentCopula, self).__init__(dim=dim)
		self.df = df
		self.sigma = sigma
		
	def getFreedomDegrees(self):
		return self.df
		
	def setFreedomDegrees(self, df):
		if df <= 0:
			raise ValueError("The degrees of freedom must be strictly greater than 0.")
		self.df = df
		
	def setCovariance(self, sigma):
		"""
		Set the covariance of the copula.

		Parameters
		----------
		sigma : numpy array (of size copula dimensions * copula dimension)
			The definite positive covariance matrix. Note that you should check yourself if the matrix is definite positive.
		"""
		S = np.asarray(sigma)
		if len(S.shape) > 2:
			raise ValueError("2-dimensional array expected, get {0}-dimensional array.".format(len(S.shape)))
		if S.shape[0] != S.shape[1]:
			raise ValueError("Covariance matrix must be a squared matrix of dimension {0}".format(self.dim))
		if len([ 1 for i in range(S.shape[0]) if S[i, i] <= 0]) > 0:
			raise ValueError("Null or negative variance encountered in covariance matrix.")
		if not(np.array_equal(np.transpose(S), S)):
			raise ValueError("Covariance matrix is not symmetric.")
		self.sigma = S
		
	def getCovariance(self):
		return self.sigma
		
	def cdf(self, x):
		self._checkDimension(x)
		tv = np.asarray([ scipy.stats.t.ppf(u, df=self.df) for u in x ])
		print(tv)
		def fun(a, b):
			return multivariate_t_distribution(np.asarray([a, b]), np.asarray([0, 0]), self.sigma, self.df, self.dim)
			
		lim_0 = lambda x: -10
		lim_1 = lambda x: tv[1]
		return fun(x[0], x[1])
		#return scipy.integrate.dblquad(fun, -10, tv[0], lim_0, lim_1)[0]
		
	def pdf(self, x):
		self._checkDimension(x)
		D = sqrtm(np.diag(np.diag(self.sigma)))
		Dinv = inv(D)
		P = np.dot(np.dot(Dinv, self.sigma), Dinv)
		
		tv = np.asarray([ scipy.stats.t.ppf(u, df=self.df) for u in x ])
		prod = 1
		for i in range(self.dim):
			prod *= scipy.stats.t.pdf(tv[i], df=self.df)
		return multivariate_t_distribution(tv, 0, P, self.df, self.dim) / prod
