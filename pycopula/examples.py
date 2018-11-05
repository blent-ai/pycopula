import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from copula import GaussianCopula
from mpl_toolkits.mplot3d import Axes3D
from pycopula.visualization import pdf_2d, cdf_2d

# The Clayton copula
clayton = GaussianCopula(dim=2)

# Visualization of CDF and PDF
u, v, C = cdf_2d(clayton)
u, v, c = pdf_2d(clayton)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d', title="Clayton copula CDF")
X, Y = np.meshgrid(u, v)

ax.set_zlim(0, 5)
ax.plot_surface(X, Y, c, cmap=cm.Blues)
ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)

ax = fig.add_subplot(122, title="Clayton copula PDF")
ax.contour(X, Y, c, levels = np.arange(0,5,0.15))

plt.show()
