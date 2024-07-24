import numpy as np
from transformer import Transformer

class PredictionGrid():
    def __init__(self, transformer, count_i=50, count_l=50, count_a=50, factor_i=1, factor_l=1, factor_a=1):
        self.transformer = transformer
        self.factor_i = factor_i
        self.factor_l = factor_l
        self.factor_a = factor_a
        self.i_grid = np.linspace(0, factor_i, count_i)
        self.l_grid  = np.linspace(0, factor_l, count_l)
        self.a_grid = np.linspace(0, factor_a, count_a)
        return
    
    def grid_for_prediction(self):
        self.X, self.Y, self.Z = np.meshgrid(self.i_grid, self.l_grid, self.a_grid)
        return np.array([self.X.flatten(), self.Y.flatten(), self.Z.flatten()]).T     
    
    def grid_for_plotting(self):
        i_grid = self.transformer.reverse_transform_i(self.i_grid, factor=self.factor_i)
        l_grid = self.transformer.reverse_transform_l(self.l_grid, factor=self.factor_l)
        a_grid = self.transformer.reverse_transform_a(self.a_grid, factor=self.factor_a)
        return i_grid, l_grid, a_grid
    
    def custom_grid_for_prediction(self, s):
        i_grid = self.transformer.transform_i(np.logspace(s.i_min, s.i_max, s.i_count))
        l_grid = self.transformer.transform_l(np.logspace(np.log10(s.l_min), np.log10(s.l_max), s.l_count))
        a_grid = self.transformer.transform_a(np.linspace(s.a_min, s.a_max, s.a_count))
        self.X, self.Y, self.Z = np.meshgrid(i_grid, l_grid, a_grid)
        return np.array([self.X.flatten(), self.Y.flatten(), self.Z.flatten()]).T
   
    def custom_grid_for_plotting(self, s):
        i_grid = np.logspace(s.i_min, s.i_max, s.i_count)
        l_grid = np.logspace(np.log10(s.l_min), np.log10(s.l_max), s.l_count)
        a_grid = np.linspace(s.a_min, s.a_max, s.a_count)
        return i_grid, l_grid, a_grid
    
    
        
    