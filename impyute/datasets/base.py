import numpy as np
from impyute.datasets.missing import Corruptor


def random_uniform(bound=(0, 10), shape=(5, 5), missingness="mcar",
                   thr=0.2, dtype="int"):

    if dtype == "int":
        data = np.random.randint(bound[0], bound[1], size=shape).astype(float)
    elif dtype == "float":
        data = np.random.uniform(bound[0], bound[1], size=shape)
    corruptor = Corruptor(data, thr=thr)
    raw_data = getattr(corruptor, missingness)()
    return raw_data


def random_normal(theta=(0, 1), shape=(5, 5), missingness="mcar", thr=0.2,
                  dtype="float"):

    mean, sigma = theta
    data = np.random.normal(mean, sigma, size=shape)
    if dtype == "int":
        data = np.round(data)
    elif dtype == "float":
        pass
    corruptor = Corruptor(data, thr=thr)
    raw_data = getattr(corruptor, missingness)()
    return raw_data


def custom_data(mask=np.zeros((3, 3), dtype=bool)):

    shape = np.shape(mask)
    data = np.reshape(np.arange(np.product(shape)), shape).astype("float")
    data[mask] = np.nan
    return data


def mnist(missingness="mcar", thr=0.2):

    from sklearn.datasets import fetch_mldata
    dataset = fetch_mldata('MNIST original')
    corruptor = Corruptor(dataset.data, thr=thr)
    data = getattr(corruptor, missingness)()
    return {"X": data, "Y": dataset.target}