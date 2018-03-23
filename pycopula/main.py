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

clayton = ArchimedeanCopula(family="amh", dim=2)
indep = Copula(dim=2, name='frechet_up')
student = StudentCopula(dim=2)
gaussian = GaussianCopula(dim=2)

#clayton.fit(data, method='cmle')
gaussian.fit(data)

u, v, carchi = pdf_2d(clayton, zclip=5)
u, v, Carchi = cdf_2d(clayton)
u, v, Cgauss = cdf_2d(gaussian)
u, v, cgauss = pdf_2d(gaussian, zclip=5)
u, v, Cstudent = cdf_2d(student)
u, v, cstudent = pdf_2d(student)
#sys.exit()
print(clayton)
print(indep)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d', title="Student copula CDF")
X, Y = np.meshgrid(u, v)

#c[c>5]= np.nan
ax.set_zlim(0, 1)
#ax.set_zlim(0, 8)

ax.plot_surface(X, Y, Cstudent, cmap=cm.Blues)
ax.plot_wireframe(X, Y, Cstudent, color='black', alpha=0.3)

ax = fig.add_subplot(122, projection='3d', title="Student copula PDF")
X, Y = np.meshgrid(u, v)

ax.set_zlim(0, 5)
ax.plot_surface(X, Y, cstudent, cmap=cm.Blues)
ax.plot_wireframe(X, Y, cstudent, color='black', alpha=0.3)

ax = fig.add_subplot(122, title="Student copula PDF")

ax.contour(X, Y, cstudent, levels = np.arange(0,5,0.15))

gaussian.setCovariance([[1, 0.8], [0.8, 1]])
clayton.setParameter(0.85)
sim = simulate(student, 3000)

fig = plt.figure()
plt.contour(X, Y, cstudent, levels = np.arange(0,5,0.15), alpha=0.4)
plt.scatter([ s[0] for s in sim ], [s[1] for s in sim ], alpha=0.4, edgecolors='none')
plt.title("Simulation of 1000 points with Clayton copula")
plt.xlim(0, 1)
plt.ylim(0, 1)

downI, upI, tailDown, tailUp = concentrationFunction(sim)
ClaytonDown = [ student.concentrationDown(x) for x in downI ]
ClaytonUp = [ student.concentrationUp(x) for x in upI ]

plt.figure()
plt.plot(downI, tailDown, color='red', linewidth=3, label="Empirical concentration")
plt.plot(upI, tailUp, color='red', linewidth=3)
plt.plot(downI, ClaytonDown, color='blue', linewidth=1, label="Clayton concentration")
plt.plot(upI, ClaytonUp, color='blue', linewidth=1)
plt.plot([0.5, 0.5], [0, 1], color='gray', alpha=0.6, linestyle='--', linewidth=1)
plt.title("Lower-Upper tail dependence Coefficients")
plt.xlabel("Lower Tail      Upper Tail")
plt.legend(loc=0)
plt.show()
