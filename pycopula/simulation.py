# Theory : https://people.math.ethz.ch/~embrecht/ftp/copchapter.pdf

import numpy as np
import scipy.stats as stats
from scipy.linalg import sqrtm
from numpy.linalg import inv, cholesky
from scipy.stats import multivariate_normal, invgamma, t as student
import math

def simulate(copula, n):
	"""
	Generates random variables with selected copula's structure.

	Parameters
	----------
	copula : Copula
		The Copula to sample.
	n : integer
		The size of the sample.
	"""
	d = copula.getDimension()
	
	X = []
	if type(copula).__name__ == "GaussianCopula":
		# We get correlation matrix from covariance matrix
		Sigma = copula.getCovariance()
		D = sqrtm(np.diag(np.diag(Sigma)))
		Dinv = inv(D)
		P = np.dot(np.dot(Dinv, Sigma), Dinv)
		A = cholesky(P)
		
		for i in range(n):
			Z = np.random.normal(size=d)
			V = np.dot(A, Z)
			U = stats.norm.cdf(V)
			X.append(U)
	elif type(copula).__name__ == "ArchimedeanCopula":
		U = np.random.rand(n, d)
		
		# LaplaceâStieltjes invert transform
		LSinv = { 'clayton' : lambda theta: np.random.gamma(shape=1./theta), 
				'gumbel' : lambda theta: stats.levy_stable.rvs(1./theta, 1., 0, math.cos(math.pi / (2 * theta))**theta), 
				'frank' : lambda theta: stats.logser.rvs(1. - math.exp(-theta)), 
				'amh' : lambda theta: stats.geom.rvs(theta)}

		for i in range(n):
			V = LSinv[copula.getFamily()](copula.getParameter())
			X_i = [ copula.inverseGenerator(-np.log(u) / V) for u in U[i, :] ]
			X.append(X_i)
	elif type(copula).__name__ == "StudentCopula":
		nu = copula.getFreedomDegrees()
		Sigma = copula.getCovariance()

		for i in range(n):
			Z = multivariate_normal.rvs(size=1, cov=Sigma)
			W = invgamma.rvs(nu / 2., size=1)
			U = np.sqrt(W) * Z
			X_i = [ student.cdf(u, nu) for u in U ]
			X.append(X_i)

	return X
