#!/usr/bin/env python

from setuptools import setup

with open("README.md", 'r') as f:
	long_description = f.read()

with open('requirements.txt', 'r') as f:
	required = f.read().splitlines()

setup(name='PyCopula',
      version='0.1',
      description='Python copulas library for dependency modeling',
      author='Maxime Jumelle',
      author_email='maxime@aipcloud.io',
      license="Apache 2",
      long_description=long_description,
      url='https://github.com/MaximeJumelle/PyCopula/',
      install_requires=required
     )
