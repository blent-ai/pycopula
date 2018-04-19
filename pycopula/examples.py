import sys

sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pycopula.copula import *
from mpl_toolkits.mplot3d import Axes3D
from pycopula.visualization import pdf_2d
from pycopula.simulation import simulate

# The Gaussian copula withb specified covariance matrix
gaussian = GaussianCopula(dim=2, sigma=[[1, 0.8], [0.8, 1]])
# Visualization of CDF and PDF
u, v, c = pdf_2d(gaussian)
X, Y = np.meshgrid(u, v)

# Sampling of size 1000 with Gaussian copula
sim = simulate(gaussian, 1000)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, title="Gaussian copula PDF")
ax.contour(X, Y, c, levels = np.arange(0,5,0.15), alpha=0.5)
plt.scatter([ s[0] for s in sim ], [s[1] for s in sim ], alpha=0.6, edgecolors='none')
plt.title("Simulation of 1000 points with Gaussian copula")
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.show()
