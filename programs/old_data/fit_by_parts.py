import numpy as np
from fit_exp_jacquelin import FitExp

class fit_by_parts():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def fit(self):
        params = FitExp(self.x, self.y, exp_count=1, constant=True).fit()
        
        
if __name__ == "__main__":
   pass 