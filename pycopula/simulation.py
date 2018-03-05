import numpy as np
import scipy.stats as stats
from scipy.linalg import sqrtm
from numpy.linalg import inv, cholesky
import math

# Only for normal copula
def simulate(copula, n):
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
		
		# Laplace–Stieltjes invert transform
		LSinv = { 'clayton' : lambda theta: np.random.gamma(shape=1./theta), 
				'gumbel' : lambda theta: stats.levy_stable.rvs(1./theta, 1., 0, math.cos(math.pi / (2 * theta))**theta) }

		for i in range(n):
			V = LSinv[copula.getFamily()](copula.getParameter())
			X_i = [ copula.inverseGenerator(-np.log(u) / V) for u in U[i, :] ]
			X.append(X_i)

	return X
