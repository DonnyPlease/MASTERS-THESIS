import sys, os
IMPORT_PATH = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(IMPORT_PATH)

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset.Dataset import DatasetUtils
from draw_dataset import draw

# Load the data

dataset, _ = DatasetUtils.load_datasets_to_dicts('dataset')

data = DatasetUtils.dataset_to_dict(dataset)

data = data["1e17"]

x = np.array([[item[0], item[1]] for item in data])
y = np.array([float(item[2]) for item in data])

# Preprocess the data
ls = [item[0] for item in data]
alphas = [item[1] for item in data]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 2. Normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=[2]),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(x_train, y_train, epochs=100)

# Evaluate the model
test_loss = model.evaluate(x_test, y_test)

print('Test loss:', test_loss)

# Make predictions
y_pred = model.predict(x_test)

plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# Create a grid for the plot
z = model.predict(scaler.transform(x))
print(z)

draw(ls, alphas, z[:,0])
