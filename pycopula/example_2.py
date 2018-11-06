import sys

sys.path.insert(0, '..')

import pandas as pd

from pycopula.visualization import pdf_2d, cdf_2d, concentrationFunction
from pycopula.simulation import simulate
from pycopula.copula import ArchimedeanCopula, GaussianCopula

data = pd.read_csv("mydata.csv").values[:,1:]

print("Begin")
archimedean = ArchimedeanCopula(family="gumbel", dim=2)
archimedean.fit(data, method="mle")
print("End")
print(archimedean)

