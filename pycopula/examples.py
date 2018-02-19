import sys

sys.path.insert(0, '..')

from pycopula.copula import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib import cm

data = pd.read_csv("mydata.csv").values[:,1:]
#plt.figure()
#plt.scatter(data[:,0], data[:,1], marker="x")
#plt.show()
#print(data.shape[1])
#print(data)

copulas = []
copulas.append(ArchimedeanCopula(family="clayton", dim=2))
copulas.append(Copula(dim=2, name='frechet_up'))
copulas.append(Copula(dim=2, name='indep'))
copulas.append(GaussianCopula(dim=2))

names = [ 'Clayton copula', 'Fréchet-Hoeffding upper bound', 'Independency copula', 'Gaussian copula' ] 

fig = plt.figure()
index = 1

for c, i in zip(copulas, range(1,5)):
	u, v, C = c.cdf_2d()

	ax = fig.add_subplot(220 + i, projection='3d', title=names[i-1])
	X, Y = np.meshgrid(u, v)
	ax.set_zlim(0, 1)
	ax.plot_surface(X, Y, C, cmap=cm.Blues)
	ax.plot_wireframe(X, Y, C, color='black', alpha=0.3)
	ax.view_init(30, 260)

plt.show()
