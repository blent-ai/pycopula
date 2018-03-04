import sys

sys.path.insert(0, '..')

from pycopula.copula import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib import cm

from pycopula.visualization import pdf_2d, cdf_2d, concentrationFunction
from pycopula.simulation import simulate

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

u, v, carchi = pdf_2d(clayton, zclip=5)
u, v, Carchi = cdf_2d(clayton)
u, v, Cgauss = cdf_2d(gaussian)
u, v, cgauss = pdf_2d(gaussian, zclip=5)

#sys.exit()
print(clayton)
print(indep)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d', title="Clayton copula CDF")
X, Y = np.meshgrid(u, v)

#c[c>5]= np.nan
ax.set_zlim(0, 1)
#ax.set_zlim(0, 8)

ax.plot_surface(X, Y, Carchi, cmap=cm.Blues)
ax.plot_wireframe(X, Y, Carchi, color='black', alpha=0.3)

ax = fig.add_subplot(122, projection='3d', title="Clayton copula PDF")
X, Y = np.meshgrid(u, v)

ax.set_zlim(0, 5)
ax.plot_surface(X, Y, cgauss, cmap=cm.Blues)
ax.plot_wireframe(X, Y, cgauss, color='black', alpha=0.3)

ax = fig.add_subplot(122, title="Clayton copula PDF")

ax.contour(X, Y, carchi, levels = np.arange(0,5,0.15))

gaussian.setCovariance([[1, 0.8], [0.8, 1]])
sim = simulate(gaussian, 1000)

fig = plt.figure()
plt.contour(X, Y, cgauss, levels = np.arange(0,5,0.15), alpha=0.4)
plt.scatter([ s[0] for s in sim ], [s[1] for s in sim ])
plt.title("Simulation of 1000 points with Gaussian copula")
plt.xlim(0, 1)
plt.ylim(0, 1)

downI, upI, tailDown, tailUp = concentrationFunction(sim)
GaussDown = [ gaussian.concentrationDown(x) for x in downI ]
GaussUp = [ gaussian.concentrationUp(x) for x in upI ]

plt.figure()
plt.plot(downI, tailDown, color='red', linewidth=3, label="Empirical concentration")
plt.plot(upI, tailUp, color='red', linewidth=3)
plt.plot(downI, GaussDown, color='blue', linewidth=1, label="Gaussian concentration")
plt.plot(upI, GaussUp, color='blue', linewidth=1)
plt.plot([0.5, 0.5], [0, 1], color='gray', alpha=0.6, linestyle='--', linewidth=1)
plt.title("Lower-Upper tail dependence Coefficients")
plt.xlabel("Lower Tail      Upper Tail")
plt.legend(loc=0)
plt.show()
