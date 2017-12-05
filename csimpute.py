from impyute.datasets import random_uniform
# from impyute.imputations.cs import random_imputation
# from impyute.imputations.cs import mean_imputation
from impyute.imputations.cs import em

raw_data = random_uniform(shape=(8, 8), missingness="mcar", th=0.2)
print(raw_data)
complete_data = em(raw_data)
print(complete_data)