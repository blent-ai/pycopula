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

def pdf_2d(copula,  step=40, zclip=None):
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
	
def cdf_2d(copula,  step=40):
	u = np.linspace(1e-4, 1.-1e-4, num=step)
	v = np.linspace(1e-4, 1.-1e-4, num=step)
	C = []

	for i in range(len(u)):
		row = []
		for j in range(len(v)):
			row.append(copula.cdf([ u[i], v[j] ]))
		C.append(row)

	return u, v, np.asarray(C)
	
