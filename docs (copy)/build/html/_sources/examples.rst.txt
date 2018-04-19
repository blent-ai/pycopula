Examples
===============================

3D Visualization
----------------

.. code-block:: python
   :emphasize-lines: 3,5

	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib import cm
	from pycopula.copula import *
	from mpl_toolkits.mplot3d import Axes3D
	from pycopula.visualization import cdf_2d

	# Storing the copulas in array
	copulas = []
	copulas.append(ArchimedeanCopula(family="clayton", dim=2))
	copulas.append(Copula(dim=2, name='frechet_up'))
	copulas.append(Copula(dim=2, name='frechet_down'))
	copulas.append(GaussianCopula(dim=2))

	names = [ 'Clayton copula', 'Fréchet-Hoeffding upper bound', 'Fréchet-Hoeffding lower bound', 'Gaussian copula' ] 

	fig = plt.figure()
	index = 1

	# For each copula
	for c, i in zip(copulas, range(1,5)):
		# We get the CDF values
		u, v, C = cdf_2d(c)
	
		# Subplotting the current copula's CDF
		ax = fig.add_subplot(220 + i, projection='3d', title=names[i-1])
		X, Y = np.meshgrid(u, v)
		ax.set_zlim(0, 1)
		ax.plot_surface(X, Y, C, cmap=cm.Blues)
		ax.plot_wireframe(X, Y, C, color='black', alpha=0.3)
		ax.view_init(30, 260)

	plt.show()
