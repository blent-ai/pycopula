import sys
import time

sys.path.insert(0, '..')

import scipy
import pandas as pd

from pycopula.visualization import pdf_2d, cdf_2d, concentrationFunction
from pycopula.simulation import simulate
from pycopula.copula import ArchimedeanCopula, GaussianCopula

import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("mydata.csv").values[:,1:]

print("Begin")
archimedean = GaussianCopula(dim=2)
elapsedTime = time.time()
archimedean.fit(data, method="cmle")
elapsedTime = time.time() - elapsedTime
print(elapsedTime)


u, v, C = cdf_2d(archimedean)
u, v, c = pdf_2d(archimedean)

fig = plt.figure()
ax = fig.add_subplot(121, title="Clayton copula CDF")
X, Y = np.meshgrid(u, v)


ax = fig.add_subplot(122, title="Clayton copula PDF")
ax.contour(X, Y, c, levels = np.arange(0,5,0.2))

plt.show()

sys.exit()
print("End")
print(archimedean)

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
