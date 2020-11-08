import numpy as np
import sklearn as sl
import sklearn.linear_model as slm
from sklearn.metrics import r2_score
import statsmodels as sm
import scipy as sp
import math
import random
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def sparse(size, proba):
    """Création d'un vecteur sparse avec une proportion de valeurs non nulles égale à proba
    size est la taille du vecteur"""
    v = []  # vecteur de valeurs
    for i in range(size):
        r = random.random()
        if r < proba:
            v.append(random.randint(1, 10))
        else:
            v.append(0)
    # sp.sparse.isspmatrix(csr_matrix(v))
    return v


class Simulation:
    """Cette classe nous permet de créer la simulation"""

    def __init__(self, n, D, t, s, num):
        """On crée les matrices X, lambda0 (qui est sparse) et epsilon (vecteur de bruit non gaussien), puis on calcule Y."""

        # prérequis
        if D <= n:
            D = n + 100  # pour que D soit très grand devant n
        if s >= D:
            s = D - 50  # s << D
            if s <= 0:
                s = 1

        # num indique avec quelle régression on va comparer la médiane (si num = 1 lasso, si 2 ridge, 3 elatic-net, 4 linéaire)
        self.num = num
        self.n = n  # nombre de lignes dans X
        self.D = D  # nombre de colonnes dans X et dans lambda0
        self.s = s  # permet de connaître la proportion de valeurs non nulles dans lambda0

        self.t = t  # nombre positif fixé
        self.k = math.floor(3.5 * t) + 1
        self.m = math.floor(self.n / self.k)

        # Construction des matrices
        self.X = np.random.rand(n, D)  # X

        proba = self.s / self.D  # proportion de valeurs non nulles dans lambda0
        self.lambda0 = np.array(sparse(self.D, proba))  # lambda0

        self.eps = np.random.rand(n)

        self.Y = []
        for i in range(n):
            c = 0
            for j in range(D):
                c += self.lambda0[j]*self.X[i, j]
            self.Y.append(c + self.eps[i])
        self.Y = np.array(self.Y)

        # les deux paramètres suivants sont construits à partir de la fonction build :
        # self.Xl
        # self.Yl

    def build(self):
        """ Dans cette fonction on construit Xl et Yl et on calcule la matrice L
        regroupant toutes les estimations Lasso de Xl et Yl """

        self.L = []  # on initialise L à une vecteur vide
        # Dans L, on va stocker les valeurs de l'estimation Lasso

        for l in range(1, self.k):

            j1 = (l - 1) * self.m + 1  # premier élément de Gl
            jm = l * self.m  # dernier élément de Gl

            # Construction de Xl et Yl
            self.Xl = self.X[j1:jm, :]  # Xl
            self.Yl = self.Y[j1: jm]  # Yl

            # on calcule les prédictions du lasso qu'on
            self.lasso = self.lasso_estimator().tolist()
            # transforme en liste pour les stocker dans un tableau

            # on construit notre tablea comme une liste de liste
            self.L.append(self.lasso)

        self.L = np.array(self.L)  # tableau des valeurs de l'estimation Lasso

    def lasso_estimator(self):
        """Cette fonction met en place le calcul de l'estimation Lasso que l'on
        stocke dans la matrice L (dans la fonction build())"""

        alpha = 0.01
        lasso = slm.Lasso(alpha=alpha, fit_intercept=True, max_iter=1000)
        clf = lasso.fit(self.Xl, self.Yl)
        pred = clf.predict(self.X)
        return pred

    def quantile_regression(self):
        """Cette fonction calcule la régression quantile sur les données estimées par la régression Lasso et affiche son R^2"""
        quantile = sm.regression.quantile_regression.QuantReg(
            self.Y, np.transpose(self.L), intercept=True)
        clq = quantile.fit(q=0.5)
        pred = clq.predict(np.transpose(self.L))

        r2_score_quantreg = 1 - r2_score(self.Y, pred)
        print("Le R^2 de la régression quantile est %f" % r2_score_quantreg)
        return pred

    def comparaison(self):
        """Cette fonction permet de comparer la régression obtenue par la régression quantile sur les estimateurs obtenus 
        avec la régression Lasso avec des estimateurs naturels. Pour rappel,
        si num = 1 -> régression lasso
        si num = 2 -> régression ridge
        si num = 3 -> régression elastic net
        si num = 4 -> régression linéaire.
        On compare les estimateurs par une mesure du R^2."""
        self.qr = self.quantile_regression()
        alpha = 0.01

        if self.num == 1:
            lasso = slm.Lasso(alpha=alpha, fit_intercept=True, max_iter=1000)
            clf = lasso.fit(self.X, self.Y)
            res = clf.predict(self.X)
            r2_score_lasso = r2_score(self.Y, res)
            print("Le R^2 de la régression lasso est de %f" % r2_score_lasso)

        if self.num == 2:
            ridge = slm.Ridge(alpha=alpha, fit_intercept=True, max_iter=1000)
            clr = ridge.fit(self.X, self.Y)
            res = clr.predict(self.X)
            r2_score_ridge = r2_score(self.Y, res)
            print("Le R^2 de la régression ridge est de %f" % r2_score_ridge)

        if self.num == 3:
            elnet = slm.ElasticNet(
                alpha=alpha, fit_intercept=True, max_iter=1000)
            clt = elnet.fit(self.X, self.Y)
            res = clt.predict(self.X)
            r2_score_elnet = r2_score(self.Y, res)
            print("Le R^2 de la régression élastic-net est de %f" %
                  r2_score_elnet)

        if self.num == 4:
            lin = slm.LinearRegression(fit_intercept=True)
            clt = lin.fit(self.X, self.Y)
            res = lin.predict(self.X)
            r2_score_lin = r2_score(self.Y, res)
            print("Le R^2 de la régression linéaire est de %f" % r2_score_lin)


def main():
    """Cette fonction permet le lancement de la simulation et execute toutes les fonctions."""
    s = Simulation(100, 1000, 3, 50, 4)  # on peut le modifier
    s.build()
    s.quantile_regression()
    s.comparaison()


# On appelle le main
main()