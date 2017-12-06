import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas import datetime

Location = r'/home/ubuntu/Downloads/pandas_plot_timeserise/USDT_BTC.csv'
# df = pd.read_csv(Location, delimiter=';', header=0, names=['date', 'high', 'low', 'close', 'volume', 'quote_volume', 'weighted_average'])
df = pd.read_csv(Location, error_bad_lines=False)

# x = df.head.date
# y = df.head.weighted_average

x = df.head(100).date
y = df.head(100).weighted_average
print(y)

x = [datetime.strptime(elem, '%Y-%m-%d %H:%M:%S') for elem in x]
(fig, ax) = pyplot.subplots(1, 1)
ax.plot(x, y)
pyplot.show()