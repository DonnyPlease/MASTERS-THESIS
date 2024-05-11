import sys, os
IMPORT_PATH = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(IMPORT_PATH)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from sklearn.svm import SVR

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=11)

scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [1,5,10,50,100,150,300,500, 700,1000],
    'gamma': [5,4.5,4,3,2,1,0.1,10],
    'kernel': ['rbf']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(x_train, y_train)

# Get the best parameters found by GridSearchCV
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model
train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)
print('Train R2 Score:', train_score)
print('Test R2 Score:', test_score)

# Make predictions
y_pred = best_model.predict(x_test)

# Plot predictions
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

num_points = 40
x_grid = np.logspace(np.log10(np.min(ls)), np.log10(np.max(ls)), num_points)
y_grid = np.linspace(np.min(alphas), np.max(alphas), num_points)
new_x = np.array([[i, j] for i in x_grid for j in y_grid])
X,Y = np.meshgrid(x_grid, y_grid)
print(new_x.shape)

# Create a grid for the plot
z = best_model.predict(scl.transform(new_x))
# print(z)
Z = griddata((new_x[:,0], new_x[:,1]), z, (X, Y), method='linear')
print(z.shape)
draw(X,Y,Z)
