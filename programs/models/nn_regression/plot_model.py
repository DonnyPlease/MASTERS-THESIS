import sys
IMPORT_PATH = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(IMPORT_PATH)
import torch
import joblib


import numpy as np
import matplotlib.pyplot as plt

from draw_dataset import draw

from pytorch_first_model import Model

if __name__ == "__main__":
    # load model
    model = torch.load("model.pth")
    scaler = joblib.load("scaler.pkl")
    model.scaler = scaler
    
    # create a grid
    num_points = 50
    l_grid = np.logspace(-2,0.75,num_points)
    alpha_grid = np.linspace(0,60,num_points)
    i_grid = np.logspace(18,18,1)
    
    X = torch.tensor(np.array([[x,y,z] for x in i_grid for z in alpha_grid for y in l_grid]), dtype=torch.float32)
    Y = model.forward(X).squeeze() # forward pass
    draw(l_grid, alpha_grid, Y.detach().numpy().reshape(num_points,num_points))
