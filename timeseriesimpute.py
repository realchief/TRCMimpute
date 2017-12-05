import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas import datetime
from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scipy.interpolate import interp1d


Location = r'/home/ubuntu/Downloads/pandas_plot_timeserise/USDT_BTC.csv'
# df = pd.read_csv(Location, delimiter=';', header=0, names=['date', 'high', 'low', 'close', 'volume', 'quote_volume', 'weighted_average'])
df = pd.read_csv(Location, error_bad_lines=False)

x = df.date
y = df.weighted_average

y_interpolated = y.interpolate()
print(y_interpolated)

x = [datetime.strptime(elem, '%Y-%m-%d %H:%M:%S') for elem in x]
(fig, ax) = pyplot.subplots(1, 1)
ax.plot(x, y_interpolated)

pyplot.show()