from impyute.datasets import random_uniform
# from impyute.imputations.cs import random_imputation
# from impyute.imputations.cs import mean_imputation
from impyute.imputations.cs import em

raw_data = random_uniform(shape=(5, 5), missingness="mcar", th=0.2)
complete_data = em(raw_data)
