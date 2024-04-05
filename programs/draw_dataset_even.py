from fit_tool.Dataset import DatasetUtils

import numpy as np
import matplotlib.pyplot as plt

dataset, _ = DatasetUtils.load_datasets_to_dicts('dataset')
data = DatasetUtils.dataset_to_dict(dataset)

current = "1e17"

# Extract the data points
x = [item[0] for item in data[current]]
y = [item[1] for item in data[current]]
z = [item[2] for item in data[current]]

# create a meshgrid
X = set(x)
Y = set(y)

X = list(X)
X.sort()

Y = list(Y)