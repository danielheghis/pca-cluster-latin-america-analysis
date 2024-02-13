'''
Clasa care incapsuleaza implementarea modelului de ACP
'''
import numpy as np

def standardise(x):
    '''
    x - data table, expect numpy.ndarray
    '''
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    Xstd = (x - means) / stds
    return Xstd


class PCA:
    # constructorul primeste o matrice X standardizata
    def __init__(self, X):

        self.X = X
        # calcul matrice de varianta-covarianta pentru X
        self.Cov = np.cov(m=X, rowvar=False)  # variabilele sunt pe coloane
        print(self.Cov.shape)

        # calcul valori proprii si vectori proprii pentru matricea de varianta-covarianta
        self.valoriProprii, self.vectoriiProprii = np.linalg.eigh(a=self.Cov)
        print(self.valoriProprii, self.valoriProprii.shape)
        print(self.vectoriiProprii.shape)

        # sortare descrescatoare valori proprii si vectori proprii
        k_desc = [k for k in reversed(np.argsort(self.valoriProprii))]
        print(k_desc)
        self.alpha = self.valoriProprii[k_desc]
        self.A = self.vectoriiProprii[:, k_desc]

        # regularizare vectorilor proprii
        for j in range(self.A.shape[1]):
            minCol = np.min(a=self.A[:, j])
            maxCol = np.max(a=self.A[:, j])
            if np.abs(minCol) > np.abs(maxCol):
                self.A[:, j] = (-1) * self.A[:, j]

        # calcul componente principale
        self.C = self.X @ self.A
        # self.C = np.matmul(self.X, self.A)  # alternativa

        # calcul corelatie dintre variabilele observate si componentele principale
        # factor loadings
        self.Rxc = self.A * np.sqrt(self.alpha)

        self.C2 = self.C * self.C
        # self.C2 = np.square(self.C)

    def getAlpha(self):
        # return self.valoriProprii
        return self.alpha

    def getA(self):
        # return self.vectoriiProprii
        return self.A

    def getComponentePrincipale(self):
        return self.C

    def getFactorLoadings(self):
        return self.Rxc

    def getScoruri(self):
        # calcul scoruri
        return self.C / np.sqrt(self.alpha)

    def getCalitateObservatii(self):
        SL = np.sum(self.C2, axis=1)  # sume pe linii
        return np.transpose(self.C2.T / SL)

    # calcul contributie observatii la explicarea variantei axelor componentelor principale
    def getContributieObservatii(self):
        return self.C2 / (self.X.shape[0] * self.alpha)

    def getComunalitati(self):
        Rxc2 = np.square(self.Rxc)
        return np.cumsum(a=Rxc2, axis=1)  # sume pe linii