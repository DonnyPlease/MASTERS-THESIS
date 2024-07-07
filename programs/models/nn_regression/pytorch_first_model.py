import sys
DATASET_FOLDER_PATH = "dataset"
IMPORT_PATH = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(IMPORT_PATH)


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset.Dataset import DatasetUtils
from draw_dataset import draw_dataset, draw
from transformer import Transformer


# create a Model class
class Model(nn.Module):
    # input layer -> hidden layer 1 -> hidden layer 2 -> output layer
    def __init__(self, input_size=3, hidden_size=32, transformer=None):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.transformer = transformer if transformer else Transformer()
       
    def scale_input(self, x):
        x = torch.tensor(self.transformer.transform(x), dtype=torch.float32)
        return x
     
    def forward(self, x):
        # x = self.scale_input(x)
        
        # forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


def load_data():
    dataset = DatasetUtils.load_final_dataset(DATASET_FOLDER_PATH)
    x, y = DatasetUtils.dataset_to_tensors(dataset)
    return x, y 

def split_data(x, y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    y = y[:,0]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = x_train.float()
    x_test = x_test.float()
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    torch.manual_seed(42)
    x, y = load_data()
    
    # transform the input data to be in the range [0,1] with log10 scaling for the second column (length)
    x = np.array(x)
    transformer = Transformer()
    transformer.fit(x)
    transformer.save('transformer.pkl')
    x = transformer.transform(x)
    
    # initialize model
    model = Model(transformer=transformer)
    x_train, x_test, y_train, y_test = split_data(x, y)
    
    # set the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Training the model
    n_epochs = 1000
    train_losses = np.zeros(n_epochs)
    for i in range(n_epochs):
        model.train()
        
        # Forward pass
        outputs = model.forward(x_train)
        outputs = outputs.squeeze()
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses[i] = loss.item()
        print(f'Epoch {i+1}/{n_epochs}, Loss: {loss.item():.4f}')
    
    # Plot the loss
    plt.plot(train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    
    outputs = model.forward(x_test).squeeze()
    loss = criterion(outputs, y_test)
    print(f'Mean Squared Error on Test Data: {loss.item():.4f}')
    
    # save the model
    torch.save(model, 'model.pth')
    joblib.dump(model.transformer, 'scaler.pkl')
    
    
    # Draw the predictions for i = 1e17
    lengths = np.logspace(np.log10(0.01), np.log10(5), 100)
    alphas = np.linspace(0, 60, 100)
    
    x = np.zeros((len(lengths)*len(alphas), 3))
    for i, l in enumerate(lengths):
        for j, a in enumerate(alphas):
            x[i*len(alphas) + j] = [1e19 8, l, a]
    print(x)        
    x = transformer.transform(x)
    x = torch.tensor(x, dtype=torch.float32)
    y_pred = model.forward(x).detach().numpy()
    print(y_pred.shape)
    # Make a grid from lengths and alphas
    # lengths, alphas = np.meshgrid(lengths, alphas)
    
    # Reshape the predictions into a grid
    y_pred = y_pred.reshape((len(lengths), len(alphas)))
    
    print(y_pred.shape)
    print(lengths.shape)
    print(alphas.shape)
    
    print(y_pred)
    print(lengths)
    print(alphas)
    
    draw(lengths, alphas, y_pred.T)
    
    