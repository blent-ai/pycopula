import numpy as np
import scipy.stats as stats
from scipy.linalg import sqrtm
from numpy.linalg import inv, cholesky

# Only for normal copula
def simulate(copula, n):
	d = copula.getDimension()
	u = np.random.rand(n, d)
	
	# We get correlation matrix from covariance matrix
	Sigma = copula.getCovariance()
	print(np.diag(np.diag(Sigma)))
	D = sqrtm(np.diag(np.diag(Sigma)))
	Dinv = inv(D)
	P = np.dot(np.dot(Dinv, Sigma), Dinv)
	A = cholesky(P)
	print(A)
	X = []
	
	for i in range(n):
		Z = np.random.normal(size=d)
		V = np.dot(A, Z)
		U = stats.norm.cdf(V)
		X.append(U)

	return X
