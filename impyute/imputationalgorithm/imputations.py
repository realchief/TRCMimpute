import random
import numpy as np
from impyute.utils import find_null
from impyute.utils import checks


class Trcmimputation:

    def __init__(self, n, ts, X_k0, P_k0):

        self.n = n
        self.ts = ts
        self.X_k0 = X_k0
        self.P_k0 = P_k0
        self.u_k1 = np.ones((n, 1))

        self.A = np.identity(n*2)
        for i in range(n):
            self.A[i][n+i] = ts
        B = np.identity(n)*((1/2)*ts**2)
        B = np.append(B, np.identity(n)*ts)
        B = np.reshape(B, (n*2, n))
        self.B = B
        self.H = np.identity((n*2))
        self.C = np.identity(n*2)

        self.R = np.ones((n*2, n*2))

        self.w_k1 = np.zeros(n*2)
        self.Q_k1 = np.zeros((self.n*2, self.n*2))
        self.Z_k1 = np.zeros((n*2, 1))

    def filter(self, Y_km):

        X_k_p = np.matmul(self.A, self.X_k0) + np.matmul(self.B,
                                                         self.u_k1) + self.w_k1

        P_k_p = np.matmul(np.matmul(self.A, self.P_k0), self.A.T) + self.Q_k1

        K = np.matmul(P_k_p, self.H.T)/(np.matmul(np.matmul(self.H, P_k_p),
                                                  self.H.T) + self.R)

        Y_k = np.matmul(self.C, Y_km) + self.Z_k1

        X_k1 = X_k_p + np.matmul(K, (Y_k - np.matmul(self.H, X_k_p)))

        P_k1 = np.matmul((np.identity(self.n*2)-np.matmul(K, self.H)), P_k_p)

        self.X_k0 = X_k1
        self.P_k0 = P_k1

    def predict_next(self):
        return self.X_k0


def random_imputation(data):

    if not checks(data):
        raise Exception("Checks failed")
    null_xy = find_null(data)
    for x, y in null_xy:
        uniques = np.unique(data[:, y])
        uniques = uniques[~np.isnan(uniques)]
        data[x][y] = np.random.choice(uniques)
    return data


def em_algorithm(data, loops=50, dtype="cont"):

    if not checks(data):
        raise Exception("Checks failed")
    if dtype == "cont":
        null_xy = find_null(data)
        for x_i, y_i in null_xy:
            col = data[:, int(y_i)]
            mu = col[~np.isnan(col)].mean()
            std = col[~np.isnan(col)].std()
            col[x_i] = random.gauss(mu, std)
            previous, i = 1, 1
            for i in range(loops):

                mu = col[~np.isnan(col)].mean()
                std = col[~np.isnan(col)].std()

                col[x_i] = random.gauss(mu, std)

                delta = (col[x_i]-previous)/previous
                if i > 5 and delta < 0.1:
                    data[x_i][y_i] = col[x_i]
                    break
                data[x_i][y_i] = col[x_i]
                previous = col[x_i]
        return data
    else:
        raise Exception("Other dtypes not supported yet.")


def from_before_observation(data, axis=0):

    if not checks(data):
        raise Exception("Checks failed")

    if axis == 0:
        data = np.transpose(data)
    elif axis == 1:
        pass

    null_xy = find_null(data)
    for x_i, y_i in null_xy:

        # Simplest scenario, look one row back
        if x_i-1 > -1:
            data[x_i][y_i] = data[x_i-1][y_i]

        # Look n rows forward

        else:
            x_residuals = np.shape(data)[0]-x_i-1  # n data points left
            val_found = False
            for i in range(1, x_residuals):
                if not np.isnan(data[x_i+i][y_i]):
                    val_found = True
                    break
            if val_found:

                for x_nan in range(i):
                    data[x_i+x_nan][y_i] = data[x_i+i][y_i]
            else:
                print("Error: Entire Column is NaN")
                raise Exception
    return data
