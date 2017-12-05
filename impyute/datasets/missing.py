import numpy as np


class Corruptor:

    def __init__(self, data, thr=0.2):
        self.dtype = data.dtype
        self.shape = np.shape(data)
        self.data = data.astype(np.float)
        self.thr = thr

    def mcar(self):
        data_1d = self.data.flatten()
        n_total = len(data_1d)
        null_x = np.random.choice(range(n_total),
                                  size=int(self.thr*n_total),
                                  replace=False)
        for x_i in null_x:
            data_1d[x_i] = np.nan
        output = data_1d.reshape(self.shape)
        return output

    def mar(self):
        """ Overwrites values with MAR placed NaN's """
        pass

    def mnar(self):
        """ Overwrites values with MNAR placed NaN's """
        pass

    def complete(self):
        """ Do nothing to the data """
        output = self.data
        return output
