#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	This file contains all the classes for copula objects.
"""

__author__ = "Maxime Jumelle"
__license__ = "Apache 2.0"
__maintainer__ = "Maxime Jumelle"
__email__ = "maxime@aipcloud.io"

from . import archimedean_generators as generators
from . import math_misc
from .math_misc import multivariate_t_distribution
from . import estimation

import numpy as np
from numpy.linalg import inv

import scipy
import scipy.misc
from scipy.stats import kendalltau, pearsonr, spearmanr, norm, t, multivariate_normal
from scipy.linalg import sqrtm
from scipy.optimize import fsolve
import scipy.integrate as integrate

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
			Default copula. 'indep' is for independency copula, 'frechet_up' the upper FrÃ©chet-Hoeffding bound and 'frechet_down' the lower FrÃ©chet-Hoeffding bound.
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

	def _check_dimension(self, x):
		if len(x) != self.dim:
			raise ValueError("Expected vector of dimension {0}, get vector of dimension {1}".format(self.dim, len(x)))
	
	def dimension(self):
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
		self._check_dimension(x)
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
		self._check_dimension(x)
		if self.name == 'indep':
			return sum([ np.prod([ x[j] for j in range(self.dim) if j != i ]) for i in range(self.dim) ])
		elif self.name in [ 'frechet_down', 'frechet_up' ]:
			raise NotImplementedError("PDF is not available for FrÃ©chet-Hoeffding bounds.")
			
	def concentration_down(self, x):
		"""
		Returns the theoritical lower concentration function.
		
		Parameters
		----------
		x : float (between 0 and 0.5)
		"""
		if x > 0.5 or x < 0:
			raise ValueError("The argument must be included between 0 and 0.5.")
		return self.cdf([x, x]) / x
		
	def concentration_up(self, x):
		"""
		Returns the theoritical upper concentration function.
		
		Parameters
		----------
		x : float (between 0.5 and 1)
		"""
		if x < 0.5 or x > 1:
			raise ValueError("The argument must be included between 0.5 and 1.")
		return (1. - 2*x + self.cdf([x, x])) / (1. - x)
		
	def concentration_function(self, x):
		"""
		Returns the theoritical concentration function.
		
		Parameters
		----------
		x : float (between 0 and 1)
		"""
		if x < 0 or x > 1:
			raise ValueError("The argument must be included between 0 and 1.")
		if x < 0.5:
			return self.concentration_down(x)
		return self.concentration_up(x)
		
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
		self.parameter = 1.5
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
		
	def inverse_generator(self, x):
		return self.generatorInvert(x, self.parameter)
		
	def get_parameter(self):
		return self.parameter
		
	def set_parameter(self, theta):
		self.parameter = theta
		
	def getFamily(self):
		return self.family

	def _check_dimension(self, x):
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
			self._check_dimension(x[0])
			return [ self.generatorInvert(sum([ self.generator(v, self.parameter) for v in row ]), self.parameter) for row in x ]
		else:
			self._check_dimension(x)
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
		self._check_dimension(x)
		# prod is the product of the derivatives of the generator for each variable
		prod = 1
		# The sum of generators that will be computed on the invert derivative
		sumInvert = 0
		# The future function (if it exists) corresponding to the n-th derivative of the invert
		invertNDerivative = None

		# Exactly 0 causes instability during computing for these copulas
		if self.family in [ "frank", "clayton"] and theta == 0:
			theta = 1e-8

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
				if theta * t < -1:
					return -theta**(order - 1) * product 
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
		self._check_dimension(X[0,:])
		estimationData = None
		
		# Moments method (only when dimension = 2)
		if method == 'moments':
			if self.kendall == None:
				self.correlations(X)
			if self.family == 'clayton':
				self.parameter = 2. * self.kendall / (1. - self.kendall)
			elif self.family == 'gumbel':
				self.parameter = 1. / (1. - self.kendall)
			elif self.family == 'frank':
				def target(x):
					return 1 - 4 / x + 4 / x**2 * integrate.quad(lambda t: t / (np.exp(t) - 1), np.finfo(np.float32).eps, x)[0] - self.kendall
				self.parameter = fsolve(target, 1)[0]
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

			is_scalar = True
			theta_start = np.array(0.5)
			bounds = theta_bounds
			if bounds == None:
				if self.family == 'amh':
					bounds = (-1, 1 - 1e-6)
					is_scalar = False
				elif self.family == 'clayton':
					bounds = (0, 10)
				elif self.family in ['gumbel', 'joe'] :
					bounds = (1, None)
					is_scalar = False

			def log_likelihood(theta):
				param_obs = np.apply_along_axis(lambda x: self.pdf_param(x, theta), arr=pobs, axis=1)
				return -np.log(param_obs).sum()
			
			if self.family == 'amh':
				theta_start = np.array(0.5)
			elif self.family in ['gumbel', 'joe']:
				theta_start = np.array(1.5)

			res = estimation.cmle(log_likelihood,
					theta_start=theta_start,
					theta_bounds=bounds,
					optimize_method=kwargs.get('optimize_method', 'Brent'),
					bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'),
					is_scalar=is_scalar)

			self.parameter = res['x'] if is_scalar else res['x'][0]
		
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
					bounds = (0, None)
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

	def __init__(self, dim=2, R=[[1, 0.5], [0.5, 1]]):
		super(GaussianCopula, self).__init__(dim=dim)
		self.set_corr(R)

	def __str__(self):
		return "Gaussian Copula :\n*Correlation : \n" + str(self.R)

	def cdf(self, x):
		self._check_dimension(x)
		return multivariate_normal.cdf([ norm.ppf(u) for u in x ], cov=self.R)

	def set_corr(self, R):
		"""
		Set the Correlation matrix of the copula.

		Parameters
		----------
		R : numpy array (of size copula dimensions * copula dimension)
			The definite positive correlation matrix. Note that you should check yourself if the matrix is definite positive.
		"""
		S = np.asarray(R)
		if len(S.shape) > 2:
			raise ValueError("2-dimensional array expected, get {0}-dimensional array.".format(len(S.shape)))
		if S.shape[0] != S.shape[1]:
			raise ValueError("Correlation matrix must be a squared matrix of dimension {0}".format(self.dim))
		if not(np.array_equal(np.transpose(S), S)):
			raise ValueError("Correlation matrix is not symmetric.")
		self.R = S
		self._R_det = np.linalg.det(S)
		self._R_inv = np.linalg.inv(S)
		
	def get_corr(self):
		return self.R

	def pdf(self, x):
		self._check_dimension(x)
		u_i = norm.ppf(x)
		return self._R_det**(-0.5) * np.exp(-0.5 * np.dot(u_i, np.dot(self._R_inv - np.identity(self.dim), u_i)))
		
	def pdf_param(self, x, R):
		self._check_dimension(x)
		if self.dim == 2 and not(hasattr(R, '__len__')):
			R = [R]
		if len(np.asarray(R).shape) == 2 and len(R) != self.dim:
			raise ValueError("Expected covariance matrix of dimension {0}.".format(self.dim))
		u = norm.ppf(x)
		
		cov = np.ones([ self.dim, self.dim ])
		idx = 0
		if len(np.asarray(R).shape) <= 1:
			if len(R) == self.dim * (self.dim - 1) / 2:
				for j in range(self.dim):
					for i in range(j + 1, self.dim):
						cov[j][i] = R[idx]
						cov[i][j] = R[idx]
						idx += 1
			else:
				raise ValueError("Expected covariance matrix, get an array.")
		
		if self.dim == 2:
			RDet = cov[0][0] * cov[1][1] - cov[0][1]**2
			RInv = 1. / RDet * np.asarray([[ cov[1][1], -cov[0][1]], [ -cov[0][1], cov[0][0] ]])
		else:
			RDet = np.linalg.det(cov)
			RInv = np.linalg.inv(cov)
		return [ RDet**(-0.5) * np.exp(-0.5 * np.dot(u_i, np.dot(RInv - np.identity(self.dim), u_i))) for u_i in u ]
		
	def quantile(self,  x):
		return multivariate_normal.ppf([ norm.ppf(u) for u in x ], cov=self.R)

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
		self._check_dimension(X[0, :])

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
					RDet = S[0, 0] * S[1, 1] - rho**2
					RInv = 1. / RDet * np.asarray([[ S[1, 1], -rho], [ -rho, S[0, 0] ]])
				else:
					RDet = np.linalg.det(S)
					RInv = np.linalg.inv(S)
				
				# Log-likelihood
				lh = 0
				for i in range(n):
					cDens = RDet**(-0.5) * np.exp(-0.5 * np.dot(ICDF[i,  :], np.dot(RInv, ICDF[i,  :])))
					lh += np.log(cDens)

				return -lh

			rho_start = [ 0.0 for i in range(int(self.dim * (self.dim - 1) / 2)) ]
			res = estimation.cmle(log_likelihood,
				theta_start=rho_start, theta_bounds=None,
				optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'),
				bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
			rho = res['x']
		elif method == 'mle':
			rho_start = [ 0.0 for i in range(int(self.dim * (self.dim - 1) / 2)) ]
			res, estimationData = estimation.mle(self, X, marginals=kwargs.get('marginals', None), hyper_param=kwargs.get('hyper_param', None), hyper_param_start=kwargs.get('hyper_param_start', None), hyper_param_bounds=kwargs.get('hyper_param_bounds', None), theta_start=rho_start, optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'), bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
			rho = res['x']
			
		self.R = np.identity(self.dim)
		# We extract rho values to covariance matrix
		for i in range(self.dim - 1):
			for j in range(i + 1,  self.dim):
				self.R[i][j] = rho[i * (self.dim - 1) + j - 1]
				self.R[self.dim - i - 1][self.dim - j - 1] = self.R[i][j]
			
		# We compute the nearest semi-definite positive matrix for the covariance matrix
		self.R = math_misc.nearPD(self.R)
		self.set_corr(self.R)

class StudentCopula(Copula):
	
	def __init__(self, dim=2, df=1, R=[[1, 0], [0, 1]]):
		super(StudentCopula, self).__init__(dim=dim)
		self.df = df
		self.R = R

	def __str__(self):
		return "Student Copula :\n*\t DF : {:1.3f}".format(self.df) + "\n*\t Correlation : \n" + str(self.R)
		
	def get_df(self):
		return self.df
		
	def set_df(self, df):
		if df <= 0:
			raise ValueError("The degrees of freedom must be strictly greater than 0.")
		self.df = df
		
	def set_corr(self, R):
		"""
		Set the covariance of the copula.

		Parameters
		----------
		R : numpy array (of size copula dimensions * copula dimension)
			The definite positive covariance matrix. Note that you should check yourself if the matrix is definite positive.
		"""
		S = np.asarray(R)
		if len(S.shape) > 2:
			raise ValueError("2-dimensional array expected, get {0}-dimensional array.".format(len(S.shape)))
		if S.shape[0] != S.shape[1]:
			raise ValueError("Covariance matrix must be a squared matrix of dimension {0}".format(self.dim))
		if len([ 1 for i in range(S.shape[0]) if S[i, i] <= 0]) > 0:
			raise ValueError("Null or negative variance encountered in covariance matrix.")
		if not(np.array_equal(np.transpose(S), S)):
			raise ValueError("Covariance matrix is not symmetric.")
		self.R = S
		
	def get_corr(self):
		return self.R
		
	def cdf(self, x):
		self._check_dimension(x)
		tv = np.asarray([ scipy.stats.t.ppf(u, df=self.df) for u in x ])

		def fun(a, b):
			return multivariate_t_distribution(np.asarray([a, b]), np.asarray([0, 0]), self.R, self.df, self.dim)
			
		lim_0 = lambda x: -10
		lim_1 = lambda x: tv[1]
		return fun(x[0], x[1])
		#return scipy.integrate.dblquad(fun, -10, tv[0], lim_0, lim_1)[0]
		
	def pdf(self, x):
		self._check_dimension(x)
		
		tv = np.asarray([ scipy.stats.t.ppf(u, df=self.df) for u in x ])
		prod = 1
		for i in range(self.dim):
			prod *= scipy.stats.t.pdf(tv[i], df=self.df)
		return multivariate_t_distribution(tv, 0, self.R, self.df, self.dim) / prod

	def fit(self, X, method='cmle', df_fixed=False, verbose=True, **kwargs):
		"""
		Fits the Student copula with specified data.

		Parameters
		----------
		X : numpy array (of size n * copula dimension)
			The data to fit.
		method : str
			The estimation method to use. Default is 'cmle'.
		df_fixed : bool
			Optimizes degrees of freedom if set to False.
		verbose : bool
			Output various informations during fitting process.
		**kwargs
			Arguments of method. See estimation for more details.

		Returns
		-------
		float
			The estimated parameters of the Gaussian copula.
		"""
		print("Fitting Student copula.")
		n = X.shape[0]
		if n < 1:
			raise ValueError("At least two values are needed to fit the copula.")
		self._check_dimension(X[0, :])

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

			ICDF = []
			if df_fixed:
				ICDF = t.ppf(pobs, df=self.df)

			def log_likelihood(params):
				if df_fixed:
					nu = self.df
					rho = params
				else:
					nu = params[0]
					rho = params[1:]
				S = np.identity(self.dim)

				if df_fixed:
					t_inv = ICDF
				else:
					t_inv = t.ppf(pobs, df=nu)
				
				# We place rho values in the up and down triangular part of the covariance matrix
				for i in range(self.dim - 1):
					for j in range(i + 1,  self.dim):
						S[i][j] = rho[i * (self.dim - 1) + j - 1]
						S[self.dim - i - 1][self.dim - j - 1] = S[i][j]
				
				# Computation of det and invert matrix
				if self.dim == 2:
					RDet = S[0, 0] * S[1, 1] - rho**2
					RInv = 1. / RDet * np.asarray([[ S[1, 1], -rho], [ -rho, S[0, 0] ]])
				else:
					RDet = np.linalg.det(S)
					RInv = np.linalg.inv(S)

				D = sqrtm(np.diag(np.diag(S)))
				Dinv = inv(D)
				P = np.dot(np.dot(Dinv, S), Dinv)
				
				# Log-likelihood
				lh = 0
				for i in range(n):
					cDens = math_misc.multivariate_t_distribution(t_inv[i,  :], 0, P, nu, self.dim)
					lh += np.log(cDens)

				return -lh

			x_start = [ 0.0 for i in range(int(self.dim * (self.dim - 1) / 2)) ]
			if not(df_fixed):
				x_start = [ 1.0 ] + x_start
			
			res = estimation.cmle(log_likelihood,
				theta_start=x_start, theta_bounds=None,
				optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'),
				bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
			fitted_params = res['x']

		self.R = np.identity(self.dim)
		# We extract rho values to covariance matrix
		if df_fixed:
			nu = self.df
			rho = fitted_params
		else:
			nu = fitted_params[0]
			rho = fitted_params[1:]

		for i in range(self.dim - 1):
			for j in range(i + 1,  self.dim):
				self.R[i][j] = rho[i * (self.dim - 1) + j - 1]
				self.R[self.dim - i - 1][self.dim - j - 1] = self.R[i][j]
			
		# We compute the nearest semi-definite positive matrix for the covariance matrix
		self.R = math_misc.nearPD(self.R)
		self.set_corr(self.R)
		self.set_df(nu)