import numpy as np
from scipy.optimize import minimize

def cmle(log_lh, theta_start=0, theta_bounds=None, optimize_method='Nelder-Mead', bounded_optimize_method='SLSQP'):
	"""
	Computes the CMLE on a specified log-likelihood function.
	
	Parameters
	----------
	log_lh : Function
		The log-likelihood.
	theta_start : float
		Initial value of theta in optimization algorithm.
	theta_bounds : couple
		Allowed values of theta.
	optimize_method : str
		The optimization method used in SciPy minimization when no theta_bounds was specified.
	bounded_optimize_method : str
		The optimization method used in SciPy minimization under constraints
		
	Returns
	-------
	OptimizeResult
		The optimization result returned from SciPy.
	"""	
	if theta_bounds == None:
		return minimize(lambda x: -log_lh(x), theta_start, method = optimize_method)
	return minimize(lambda x: -log_lh(x), theta_start, method = bounded_optimize_method, bounds=[theta_bounds])
	
def mle(copula, X, marginals, hyper_param, hyper_param_start=None, hyper_param_bounds=None, theta_start=0, theta_bounds=None, optimize_method='Nelder-Mead', bounded_optimize_method='SLSQP'):
	"""
	Computes the MLE on specified data.
	
	Parameters
	----------
	copula : Copula
		The copula.
	X : numpy array (of size n * copula dimension)
		The data to fit.
	marginals : numpy array
		The marginals distributions. Use scipy.stats distributions or equivalent that requires pdf and cdf functions according to rv_continuous class from scipy.stat.
	hyper_param : numpy array
		The hyper-parameters for each marginal distribution. Use None when the hyper-parameter is unknow and must be estimated.
	hyper_param_start : numpy array
		The start value of hyper-parameters during optimization. Must be same dimension of hyper_param.
	hyper_param_bounds : numpy array
		Allowed values for each hyper-parameter.
	theta_start : float
		Initial value of theta in optimization algorithm.
	theta_bounds : couple
		Allowed values of theta.
	optimize_method : str
		The optimization method used in SciPy minimization when no theta_bounds was specified.
	bounded_optimize_method : str
		The optimization method used in SciPy minimization under constraints
		
	Returns
	-------
	OptimizeResult
		The optimization result returned from SciPy.
	"""	
	hyperParams = np.asarray(hyper_param)
	hyperStart = np.asarray(hyper_param_start)
	n, d = X.shape
	
	start_vector = np.repeat(0, d + 1)
	start_vector[0] = theta_start
	if hyper_param_start == None:
		start_vector[1:] = [ 1.0 for i in range(d) ]
		
	optiVector = []
	idx = 1
	
	for k in range(len(hyperParams)):
		for l in range(len(hyperParams[k])):
			optiVector.append(hyperParams[k][l])
			if hyper_param_start != None and hyperParams[k][l] != None:
				start_vector[idx] = hyperStart[k][l]
				idx += 1
	
	def log_lh(x):
		lh = 0
		v = [ x[0] ]
		idx = 1
		
		for i in range(len(optiVector)):
			if optiVector[i] == None:
				v.append(x[idx])
			else:
				v.append(optiVector[i])

		marginCDF = [ marginals[j].cdf(np.transpose(X)[j], v[j + 1]) for j in range(d) ]
		marginCDF = np.transpose(marginCDF)
		lh += sum([ np.log(copula.pdf_param(marginCDF[i], v[0])) for i in range(n)])
		lh += sum([ sum(np.log(marginals[j].pdf(np.transpose(X)[j], v[j + 1]))) for j in range(d) ])
		return lh
	
	if hyper_param_bounds == None:
		if theta_bounds == None:
			return minimize(lambda x: -log_lh(x), start_vector, method = optimize_method)
		else:
			optiBounds = np.vstack((np.array([theta_bounds]), np.tile(np.array([None, None]), [d, 1]) ))
			return minimize(lambda x: -log_lh(x), start_vector, method = bounded_optimize_method, bounds=optiBounds)
	else:
		if theta_bounds == None:
			optiBounds = np.vstack((np.array([None, None]), np.tile(np.array([None, None]), [d, 1]) ))
			return minimize(lambda x: -log_lh(x), start_vector, method = bounded_optimize_method, bounds=optiBounds)
		else:
			optiBounds = np.vstack((np.array([theta_bounds]), hyper_param_bounds))
			return minimize(lambda x: -log_lh(x), start_vector, method = bounded_optimize_method, bounds=optiBounds)
