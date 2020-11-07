# Imports
import numpy as np
import statsmodels as sm
import statsmodels.regression as smr
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Simulation des données
n = 1000
eps = np.random.normal(n)
X = np.random.rand(n, 1) * 5
X1 = np.random.normal(size=(n, 1)) * 1
X2 = np.random.normal(size=(n//2, 1)) * 10
X2 = np.vstack([X2, np.zeros((n//2, 1))])
eps = - np.abs(X1) + np.abs(X2)
Y = (0.5 * X + eps).ravel()

# Calcul des régressions
clr = LinearRegression()
clr.fit(X, Y)
print(clr.summary())
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
clq = sm.regression.quantile_regression.QuantReg(Y, X).fit(q=0.5)
print(clq.summary())

# Plot
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(X, Y, 'c.')
lin = clr.predict(X)
ax.plot(X, lin, 'ro', label="Régression linéaire")
qu = clq.predict(X)
ax.plot(X, qu, 'bo', label="Régression médiane")
ax.legend()
ax.set_title("Régression linéaire vs médiane")
