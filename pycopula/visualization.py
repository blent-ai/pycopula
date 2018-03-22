#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Useful methods for copula visualization.
"""

__author__ = "Maxime Jumelle"
__copyright__ = "Copyright 2018, AIPCloud"
__credits__ = "Maxime Jumelle"
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Maxime Jumelle"
__email__ = "maxime@aipcloud.io"
__status__ = "Development"

import numpy as np

def pdf_2d(copula, step=40, zclip=None):
	if zclip == None:
		zclip = 5
	if zclip <= 0:
		raise ValueError("The z-clip value must be strictly greater than 0.")
	u = np.linspace(1e-4, 1.-1e-4, num=step)
	v = np.linspace(1e-4, 1.-1e-4, num=step)
	C = []

	for i in range(len(u)):
		row = []
		for j in range(len(v)):
			if zclip != None:
				row.append(min(copula.pdf([ u[i], v[j] ]), zclip))
			else:
				row.append(copula.pdf([ u[i], v[j] ]))
		C.append(row)

	return u, v, np.asarray(C)
	
def cdf_2d(copula, step=40):
	u = np.linspace(1e-4, 1.-1e-4, num=step)
	v = np.linspace(1e-4, 1.-1e-4, num=step)
	C = []

	for i in range(len(u)):
		row = []
		for j in range(len(v)):
			row.append(copula.cdf([ u[i], v[j] ]))
		C.append(row)

	return u, v, np.asarray(C)

def concentrationFunction(X, step=50):
	data = np.asarray(X)
	(n, d) = data.shape
	downI = np.linspace(0.05+1e-4, 0.5, num=step)
	upI = np.linspace(0.5, 0.95-1e-4, num=step)
	# Pseudo-observations from real data X
	pobs = []
	for i in range(d):
		order = data[:, i].argsort()
		ranks = order.argsort()
		u_i = [ (r + 1) / (n + 1) for r in ranks ]
		pobs.append(u_i)
		
	def down(x):
		un = [ 1 for i in range(n) if pobs[0][i] <= x]
		if len(un) > 0:
			return sum([ 1 for i in range(n) if pobs[0][i] <= x and pobs[1][i] <= x]) / sum(un)
		return 0

	def up(x):
		un = [ 1 for i in range(n) if pobs[0][i] >= 1.-x]
		if len(un) > 0:
			return sum([ 1 for i in range(n) if pobs[0][i] >= 1.-x and pobs[1][i] >= 1.-x]) / sum(un)
		return 0

	return downI, upI, [ down(p) for p in downI ], [ up(p) for p in downI[::-1] ]
