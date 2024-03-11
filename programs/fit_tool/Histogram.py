import numpy as np


class Histogram():
    def __init__(self):
        self.bins = None
        self.counts = None
        self.I = None
        self.alpha = None
        self.L = None
        self.T_hot = None
        self.hot_exp_parameters = (None, None)
        self.folder_name = None
        self.jacquelin_fit = None
        self.single_fit = None
        self.left_cut = None
        self.right_cut = None
        
    def load_from_folder(self, path):
        self.bins = np.loadtxt(path+'/bins.txt')
        self.counts = np.loadtxt(path+'/counts.txt')
        self.folder_name = path
        self.set_parameters_from_folder_name()
    
    def set_parameters_from_folder_name(self):
        name = self.folder_name.split('/')[-1]
        self.I = int(name[7:9])
        self.L = int(name[10])+int(name[11:13])/100
        self.alpha = int(name.split('_')[-1])
        self.print_params()
    
    def print_params(self):
        print(self.I)
        print(self.alpha)
        print(self.L)
    
    def plot(self):
        pass
    
    def trim(self):
        pass

    def fit(self):
        pass
    
    def get_file_name(self):
        return self.file_name
    
    def get_selected_range(self):
        x = np.array(self.bins)
        if self.left_cut is None or self.right_cut is None:
            return np.array(self.bins), np.array(self.counts)
        if self.left_cut > self.right_cut:
            self.left_cut, self.right_cut = self.right_cut, self.left_cut
        condition1 = x <= self.right_cut
        condition2 = x >= self.left_cut
        condition = np.logical_and(condition1,condition2)
        return np.array(self.bins)[condition], np.array(self.counts)[condition]