import random
import numpy as np
from impyute.utils import find_null
from impyute.utils import checks


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
                # Expectation
                mu = col[~np.isnan(col)].mean()
                std = col[~np.isnan(col)].std()
                # Maximization
                col[x_i] = random.gauss(mu, std)
                # Break out of loop if likelihood doesn't change at least 10%
                # and has run at least 5 times
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
            x_residuals = np.shape(data)[0]-x_i-1  # n datapoints left
            val_found = False
            for i in range(1, x_residuals):
                if not np.isnan(data[x_i+i][y_i]):
                    val_found = True
                    break
            if val_found:
                # pylint: disable=undefined-loop-variable
                for x_nan in range(i):
                    data[x_i+x_nan][y_i] = data[x_i+i][y_i]
            else:
                print("Error: Entire Column is NaN")
                raise Exception
    return data
