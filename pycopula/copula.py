#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	This file contains all the classes for copula objects.
"""

__author__ = "Maxime Jumelle"
__copyright__ = "Copyright 2018, AIPCloud"
__credits__ = "Maxime Jumelle"
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = "Maxime Jumelle"
__email__ = "maxime@aipcloud.io"
__status__ = "Development"

import sys

sys.path.insert(0, '..')

import archimedean_generators as generators
from scipy.stats import kendalltau, pearsonr, spearmanr, norm, multivariate_normal
from scipy.optimize import minimize
from scipy.misc import factorial
import statsmodels.api as sm
import numpy as np

# An abstract copula object, which is similar to the independency copula
class Copula():

	def __init__(self, dim=2, name='indep'):
		if dim < 2 or int(dim) != dim:
			raise ValueError("Copula dimension must be an integer greater than 1.")
		self.dim = dim
		self.name = name

	def __str__(self):
		return "Copula ({0}).".format(self.name)

	def checkDimension(self, x):
		if len(x) != self.dim:
			raise ValueError("Expected vector of dimension {0}, get vector of dimension {1}".format(self.dim, len(x)))

	def cdf(self, x):
		self.checkDimension(x)
		if self.name == 'indep':
			return np.prod(x)
		elif self.name == 'frechet_up':
			return min(x)
		elif self.name == 'frechet_down':
			return max(sum(x) - self.dim + 1., 0)

	def cdf_2d(self, step=40):
		u = np.linspace(1e-4, 1.-1e-4, num=step)
		v = np.linspace(1e-4, 1.-1e-4, num=step)
		C = []

		for i in range(len(u)):
			row = []
			for j in range(len(v)):
				row.append(self.cdf([ u[i], v[j] ]))
			C.append(row)

		return u, v, np.asarray(C)

	def pdf(self, x):
		self.checkDimension(x)
		if self.name == 'indep':
			return sum([ np.prod([ x[j] for j in range(self.dim) if j != i ]) for i in range(self.dim) ])
		elif self.name in [ 'frechet_down', 'frechet_up' ]:
			raise NotImplementedError("PDF is not available for Fréchet-Hoeffding bounds.")

	def pdf_2d(self, step=40, zclip=5):
		u = np.linspace(1e-4, 1.-1e-4, num=step)
		v = np.linspace(1e-4, 1.-1e-4, num=step)
		C = []

		for i in range(len(u)):
			row = []
			for j in range(len(v)):
				z = self.pdf([ u[i], v[j] ])
				if z > zclip:
					z = zclip
				row.append(z)
			C.append(row)

		return u, v, np.asarray(C)

	#def fit(self, data):


	#def getCorrelations(self):
				
		

class ArchimedeanCopula(Copula):

	families = [ 'clayton', 'gumbel', 'frank', 'joe', 'amh' ]

	def __init__(self, family='clayton', generator=None, dim=2):
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
			if generator == None:
				raise ValueError("The generator is needed for custom archimedean copula.")
		

	def __str__(self):
		return "Archimedean Copula ({}) :".format(self.family) + "\n*\tParameter : {:1.6f}".format(self.parameter)

	def checkDimension(self, x):
		if len(x) != self.dim:
			raise ValueError("Expected vector of dimension {0}, get vector of dimension {1}".format(self.dim, len(x)))

	def cdf(self, x):
		# Gérer le cas où le générateur inverse n'est pas défini
		self.checkDimension(x)
		return self.generatorInvert(sum([ self.generator(v, self.parameter) for v in x ]), self.parameter)

	def pdf_param(self, x, theta):
		# gérer la n-ième dérivée de la réciproque de phi
		self.checkDimension(x)
		prod = 1
		sumInvert = 0
		invertNDerivative = None

		if self.family == 'clayton':
			for i in range(self.dim):
				prod *= -x[i]**(-theta - 1.)
				sumInvert += self.generator(x[i], theta)

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

				invertNDerivative = gumbelInvertDerivative
			# Fix for n dimension
		elif self.family == 'frank':
			if self.dim == 2:
				for i in range(self.dim):
					prod *= theta / (1. - np.exp(theta * x[i]))
					sumInvert += self.generator(x[i], theta)

				def frankInvertDerivative(t, theta, order):
					C = np.exp(-theta) - 1.
					return - C / theta * np.exp(t) / (C + np.exp(t))**2

				invertNDerivative = frankInvertDerivative
			# Fix for n dimension
		elif self.family == 'joe':
			if self.dim == 2:
				for i in range(self.dim):
					prod *= -theta * (1. - x[i])**(theta - 1.) / (1. - (1. - x[i])**theta)
					sumInvert += self.generator(x[i], theta)

				def joeInvertDerivative(t, theta, order):
					return 1. / theta**2 * (1. - np.exp(-t))**(1. / theta) * (theta * np.exp(t) - 1.) / (np.exp(t) - 1.)**2

				invertNDerivative = joeInvertDerivative
			# Fix for n dimension
		# Need some work on AMH
		elif self.family == 'amh':
			if self.dim == 2:
				for i in range(self.dim):
					prod *= (theta - 1.) / (theta * (x[i] - 1.) * x[i] + x[i])
					sumInvert += self.generator(x[i], theta)

				def amhInvertDerivative(t, theta, order):
					return (1. - theta) * (2. * np.exp(2. * t) / (np.exp(t) - theta)**3 - np.exp(t) / (np.exp(t) - theta)**2)

				invertNDerivative = amhInvertDerivative
			# Fix for n dimension	

		if invertNDerivative == None:
			raise Exception("The {0}-th derivative of the invert of the generator is not defined.".format(self.dim))
		return prod * invertNDerivative(sumInvert, theta, self.dim)

	def pdf(self, x):
		return self.pdf_param(x, self.parameter)

	def fit(self, X, method='cml', verbose=False, thetaBounds=None):
		n = X.shape[0]
		if n < 1:
			raise ValueError("At least two values are needed to fit the copula.")
		self.checkDimension(X[0,:])

		# Canonical Maximum Likelihood Estimation
		if method == 'cml':
			# Pseudo-observations from real data X
			pobs = []
			for i in range(self.dim):
				order = X[:,i].argsort()
				ranks = order.argsort()
				u_i = [ (r + 1) / (n + 1) for r in ranks ]
				pobs.append(u_i)

			pobs = np.transpose(np.asarray(pobs))

			bounds = thetaBounds
			if bounds == None:
				if self.family == 'amh':
					bounds = (-1, 1 - 1e-6)
				elif self.family == 'clayton':
					bounds = (-1, None)
				elif self.family in ['gumbel', 'joe'] :
					bounds = (1, None)

			def neg_log_likelihood(theta):
				return -sum([ np.log(self.pdf_param(pobs[i,:], theta)) for i in range(n) ])

			theta_start = np.array(2.)
			if self.family == 'amh':
				theta_start = np.array(0.5)

			if bounds != None:
				res = minimize(neg_log_likelihood, theta_start, method = 'SLSQP', bounds=[bounds])
			else:
				res = minimize(neg_log_likelihood, theta_start, method = 'Nelder-Mead')	
	
			self.parameter = res['x'][0]

		return self.parameter


class GaussianCopula(Copula):

	def __init__(self, dim=2, sigma=[[1, 0.5], [0.5, 1]]):
		super(GaussianCopula, self).__init__()
		self.sigma = np.asarray(sigma)

	def cdf(self, x):
		self.checkDimension(x)
		return multivariate_normal.cdf([ norm.ppf(u) for u in x ], cov=self.sigma)

	def pdf(self, x):
		self.checkDimension(x)
		detSigma = np.linalg.det(self.sigma)
		sigmaInv = np.linalg.inv(self.sigma)
		u_i = [ norm.ppf(v) for v in x ]
		return detSigma**(-0.5) * np.exp(-0.5 * np.dot(u_i, np.dot(sigmaInv - np.identity(self.dim), u_i)))

#class StudentCopula(Copula):

