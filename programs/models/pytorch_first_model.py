import sys
DATASET_FOLDER = "dataset"
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from draw_dataset import draw_dataset, draw
from dataset.Dataset import DatasetUtils
from draw_dataset import draw, draw_slice
from transformer import Transformer, transform
from models.prediction_grid import PredictionGrid

PATH_TO_MODEL = PATH_TO_PROJECT + 'models/models/nn_model.pkl'

TRAIN, TEST = 0, 1


# create a Model class
class Model(nn.Module):
    # input layer -> hidden layer 1 -> hidden layer 2 -> output layer
    def __init__(self, input_size=3, hidden_size=64):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        torch.manual_seed(42)
    
    def forward(self, x):
        # forward pass
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.out(x)
        return x


class NNModel:
    def __init__(self, transformer=None):
        self.model = Model()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1, weight_decay=1)
        self.transformer = transformer
        
    def train(self, x, y, n_epochs=1000):
        x, y = self.retype_data(x, y)
        train_losses = np.zeros(n_epochs)
        for i in range(n_epochs):
            self.model.train()
            
            # Forward pass
            outputs = self.model.forward(x)
            outputs = outputs.squeeze()
            loss = self.criterion(outputs, y)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses[i] = loss.item()
            print(f'Epoch {i+1}/{n_epochs}, Loss: {loss.item():.4f}')
            
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return non_negative(self.model.forward(x).squeeze().detach().numpy())
    
    def save(self, path):
        joblib.dump(self, path)
    
    @staticmethod
    def load(path):
        return joblib.load(path)
    
    def set_transformer(self, transformer):
        self.transformer = transformer
        return
    
    def retype_data(self, x, y):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y
    
    def scores(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().numpy()
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        # R2
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res/ss_tot
        return r2, rmse

    def mean_residuals(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().numpy()
        return np.mean(y_true - y_pred)

    def get_params(self):
        return None
    

def train_test_split_data(x, y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    y = y[:,0]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=40, shuffle=True)
    x_train = x_train.float()
    x_test = x_test.float()
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    return x_train, x_test, y_train, y_test

def load_data():
    dataset = DatasetUtils.load_final_dataset(DATASET_FOLDER)
    x, y = DatasetUtils.dataset_to_tensors(dataset)
    return x, y 

def non_negative(x):
    return np.array([max(0, t) for t in x])

if __name__ == "__main__":
    # OPTIONS
    int_factor = 1
    ACTION = TRAIN
    
    # Load the data
    x, y = DatasetUtils.load_data_for_regression(DATASET_FOLDER)
    x, y = np.array(x), np.array(y)[:,0]    # Only the first output column - T_HOT, the second is T_HOT_STDEV
    X, y, transformer = transform(x, y, factor_i=int_factor)
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=40)
    X_train, X_test, y_train, y_test = X, X, y, y
    # Train or load the model 
    nn = NNModel()
    X_train, y_train = nn.retype_data(X_train, y_train)
    X_test, y_test = nn.retype_data(X_test, y_test)
    if ACTION == TRAIN:
        nn.set_transformer(transformer)
        nn.train(X_train, y_train)
        nn.save(PATH_TO_MODEL)
    elif ACTION == TEST:
        nn = nn.load(PATH_TO_MODEL)
        
    # Test scores
    y_pred = nn.predict(X_test)
    rmse, r2 = nn.scores(y_test, y_pred)
    mean_residuals = nn.mean_residuals(y_test, y_pred)
    print(f"NN \t\t R2: {rmse:.2f} \t RMSE: {r2:.2f}")
    print(f"NN \t\t Mean of residuals: {mean_residuals:.2f}")
    
    prediction_grid = PredictionGrid(transformer=transformer, factor_i=int_factor,count_i=51)
    grid = prediction_grid.grid_for_prediction()
    grid, _ = nn.retype_data(grid, np.zeros(grid.shape[0]))
    t_hot_predicted = nn.predict(grid)
    t_hot_predicted = t_hot_predicted.reshape(prediction_grid.X.shape)
    
    # Plot the predictions 
    i_grid, l_grid, a_grid = prediction_grid.grid_for_plotting()
    slices_at = [0, 25, 50]
    for slice_at in slices_at:
        draw_slice(i_grid, l_grid, a_grid, t_hot_predicted, slice_at=slice_at, axes=["l","a"])
    
        
        
        
        
    # x, y = load_data()
    
    # # transform the input data to be in the range [0,1] with log10 scaling for the second column (length)
    # x = np.array(x)
    # transformer = Transformer()
    # transformer.fit(x)
    # transformer.save('transformer.pkl')
    # x = transformer.transform(x)
    
    # initialize model
    # model = Model(transformer=transformer)
    # x_train, x_test, y_train, y_test = split_data(x, y)
    
    # # set the loss and optimizer
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Training the model
    # n_epochs = 3500
    # train_losses = np.zeros(n_epochs)
    # for i in range(n_epochs):
    #     model.train()
        
    #     # Forward pass
    #     outputs = model.forward(x_train)
    #     outputs = outputs.squeeze()
    #     loss = criterion(outputs, y_train)
        
    #     # Backward and optimize
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
        
    #     train_losses[i] = loss.item()
    #     print(f'Epoch {i+1}/{n_epochs}, Loss: {loss.item():.4f}')
    
    # # Plot the loss
    # plt.plot(train_losses)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.show()
    
    
    # outputs = model.forward(x_test).squeeze()
    # loss = criterion(outputs, y_test)

    # print(f'Mean Squared Error on Test Data: {loss.item():.4f}')
    # print(f"Root Mean Squared Error on Test Data: {np.sqrt(loss.item()):.4f}")
    
    # # Mean of residuals
    # residuals = outputs.detach().numpy() - y_test.detach().numpy() 
    # print(f"Mean of residuals: {np.mean(residuals):.4f}")

    # # save the model
    # torch.save(model, 'nn_model.pth')
    # joblib.dump(model.transformer, 'scaler.pkl')
    
    
    # Draw the predictions for i = 1e17
    # lengths = np.logspace(np.log10(0.01), np.log10(5), 100)
    # alphas = np.linspace(0, 60, 100)
    
    # x = np.zeros((len(lengths)*len(alphas), 3))
    # for i, l in enumerate(lengths):
    #     for j, a in enumerate(alphas):
    #         x[i*len(alphas) + j] = [1e17, l, a]
     
    # x = transformer.transform(x)
    # x = torch.tensor(x, dtype=torch.float32)
    # y_pred = model.forward(x).detach().numpy()

    # # Reshape the predictions into a grid
    # y_pred = y_pred.reshape((len(lengths), len(alphas)))
    
    # draw(lengths, alphas, y_pred.T)
    
    # x = np.zeros((len(lengths)*len(alphas), 3))
    # for i, l in enumerate(lengths):
    #     for j, a in enumerate(alphas):
    #         x[i*len(alphas) + j] = [1e18, l, a]
     
    # x = transformer.transform(x)
    # x = torch.tensor(x, dtype=torch.float32)
    # y_pred = model.forward(x).detach().numpy()

    # # Reshape the predictions into a grid
    # y_pred = y_pred.reshape((len(lengths), len(alphas)))
    
    # draw(lengths, alphas, y_pred.T)
    
    # x = np.zeros((len(lengths)*len(alphas), 3))
    # for i, l in enumerate(lengths):
    #     for j, a in enumerate(alphas):
    #         x[i*len(alphas) + j] = [1e19, l, a]
     
    # x = transformer.transform(x)
    # x = torch.tensor(x, dtype=torch.float32)
    # y_pred = model.forward(x).detach().numpy()

    # # Reshape the predictions into a grid
    # y_pred = y_pred.reshape((len(lengths), len(alphas)))
    
    # draw(lengths, alphas, y_pred.T)
    
    