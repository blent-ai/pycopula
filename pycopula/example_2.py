import sys
import time

sys.path.insert(0, '..')

import scipy
import pandas as pd

from pycopula.visualization import pdf_2d, cdf_2d, concentrationFunction
from pycopula.simulation import simulate
from pycopula.copula import ArchimedeanCopula, GaussianCopula, StudentCopula

import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("mydata.csv", sep=";").values[:, 1:]
data = [ [ x.replace(",", ".") for x in row ] for row in data ]
data = np.random.normal(size=1000)
data = np.vstack((data, np.cos(data))).T
data = np.asarray(data).astype(np.float)
#data = np.hstack((data, data + np.random.rand(1000, 2)))
print(data.shape)

#np.__config__.show()
print("Begin")
archimedean = StudentCopula(dim=2)
elapsedTime = time.time()
archimedean.fit(data, method="cmle", df_fixed=False)
elapsedTime = time.time() - elapsedTime
print(elapsedTime)
print(archimedean)
sys.exit()
print("End")

clayton = ArchimedeanCopula(family="clayton", dim=2)
boundAlpha = [0, None] # Greater than 0
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
