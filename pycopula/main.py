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

clayton = ArchimedeanCopula(family="clayton", dim=2)
indep = Copula(dim=2, name='frechet_up')
gaussian = GaussianCopula(dim=2)

clayton.fit(data, method='cml')
gaussian.fit(data)

u, v, Carchi = clayton.pdf_2d()
u, v, Cindep = indep.cdf_2d()
u, v, Cgauss = gaussian.cdf_2d()
u, v, cgauss = gaussian.pdf_2d(zclip=5)

#sys.exit()
print(clayton)
print(indep)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d', title="Gaussian copula CDF")
X, Y = np.meshgrid(u, v)

#c[c>5]= np.nan
ax.set_zlim(0, 1)
#ax.set_zlim(0, 8)

ax.plot_surface(X, Y, Cgauss, cmap=cm.Blues)
ax.plot_wireframe(X, Y, Cgauss, color='black', alpha=0.3)

ax = fig.add_subplot(122, projection='3d', title="Gaussian copula PDF")
X, Y = np.meshgrid(u, v)

#c[c>5]= np.nan
ax.set_zlim(0, 5)
#ax.set_zlim(0, 8)

ax.plot_surface(X, Y, cgauss, cmap=cm.Blues)
ax.plot_wireframe(X, Y, cgauss, color='black', alpha=0.3)

plt.show()
