#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	This file contains the generators and their inverses for common archimedean copulas.
"""

import numpy as np

def boundsConditions(x):
	if x < 0 or x > 1:
		raise ValueError("Unable to compute generator for x equals to {}".format(x))

def claytonGenerator(x, theta):
	boundsConditions(x)
	if theta == 0:
		raise ValueError("The parameter of a Clayton copula must not be equal to 0.")
	if theta < -1:
		raise ValueError("The parameter of a Clayton copula must be greater than -1 and different from 0.")
	return (1. / theta) * (x**(-theta) - 1.)

def claytonGeneratorInvert(x, theta):
	if theta == 0:
		raise ValueError("The parameter of a Clayton copula must not be equal to 0.")
	if theta < -1:
		raise ValueError("The parameter of a Clayton copula must be greater than -1 and different from 0.")
	return (1. + theta * x)**(-1. / theta)

def gumbelGenerator(x, theta):
	boundsConditions(x)
	if theta < 1:
		raise ValueError("The parameter of a Gumbel copula must be greater than 1.")
	return (-np.log(x))**theta

def gumbelGeneratorInvert(x, theta):
	if theta < 1:
		raise ValueError("The parameter of a Gumbel copula must be greater than 1.")
	return np.exp(-x**(1. / theta))

def frankGenerator(x, theta):
	boundsConditions(x)
	if theta == 0:
		raise ValueError("The parameter of a Frank copula must not be equal to 0.")
	return -np.log((np.exp(-theta * x) - 1.) / (np.exp(-theta) - 1.))

def frankGeneratorInvert(x, theta):
	if theta == 0:
		raise ValueError("The parameter of a Frank copula must not be equal to 0.")
	return -1. / theta * np.log(1. + np.exp(-x) * (np.exp(-theta) - 1.))

def joeGenerator(x, theta):
	boundsConditions(x)
	if theta < 1:
		raise ValueError("The parameter of a Joe copula must be greater than 1.")
	return -np.log(1. - (1. - x)**theta)

def joeGeneratorInvert(x, theta):
	if theta < 1:
		raise ValueError("The parameter of a Joe copula must be greater than 1.")
	return 1. - (1. - np.exp(-x))**(1. / theta)

def aliMikhailHaqGenerator(x, theta):
	boundsConditions(x)
	if theta < -1 or theta >= 1:
		raise ValueError("The parameter of an Ali-Mikhail-Haq copula must be between -1 included and 1 excluded.")
	return np.log((1. - theta * (1. - x)) / x)

def aliMikhailHaqGeneratorInvert(x, theta):
	if theta < -1 or theta >= 1:
		raise ValueError("The parameter of an Ali-Mikhail-Haq copula must be between -1 included and 1 excluded.")
	return (1. - theta) / (np.exp(x) - theta)
