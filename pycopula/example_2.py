import sys

sys.path.insert(0, '..')

import scipy
import pandas as pd

from pycopula.visualization import pdf_2d, cdf_2d, concentrationFunction
from pycopula.simulation import simulate
from pycopula.copula import ArchimedeanCopula, GaussianCopula

data = pd.read_csv("mydata.csv").values[:,1:]

print("Begin")
archimedean = ArchimedeanCopula(family="clayton", dim=2)
archimedean.fit(data, method="cmle")
print("End")
print(archimedean)

clayton = ArchimedeanCopula(family="clayton", dim=2)
boundAlpha = [0, 1] # Greater than 0
boundLambda = [0, 0.5]
bounds = [ boundAlpha, boundLambda ]
paramX1 = { 'a': None, 'scale': 1.2 } # Hyper-parameters of Gamma
paramX2 = { 'scale': None } # Hyper-parameters of Exp
hyperParams = [ paramX1, paramX2 ] # The hyper-parameters
gamma = scipy.stats.gamma # The Gamma distribution
expon = scipy.stats.expon # The Exponential distribution
# Fitting copula with MLE method and Gamma/Exp marginals distributions
print(clayton.fit(data, method='ifm', marginals=[gamma, expon], hyper_param=hyperParams, hyper_param_bounds=bounds))
print(clayton)
