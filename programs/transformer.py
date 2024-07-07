import numpy as np
import joblib

class Transformer:
    def __init__(self):
        self.x = None
        self.params = {}
        return
    
    def fit(self, x):
        self.x = x
        self.params['i_min'] = np.min(x[:,0])
        self.params['i_max'] = np.max(x[:,0])
        
        self.params['l_log_min'] = np.min(np.log10(x[:,1]))
        self.params['l_log_max'] = np.max(np.log10(x[:,1]))
        
        self.params['a_min'] = np.min(x[:,2])
        self.params['a_max'] = np.max(x[:,2])
        
    def transform(self, x):
        x[:,0] = (x[:,0] - self.params['i_min'])/(self.params['i_max'] - self.params['i_min'])
        x[:,1] = np.log10(x[:,1])
        x[:,1] = (x[:,1] - self.params['l_log_min'])/(self.params['l_log_max'] - self.params['l_log_min'])
        x[:,2] = (x[:,2] - self.params['a_min'])/(self.params['a_max'] - self.params['a_min'])
        return x
    
    def reverse_transform(self, x):
        x[:,0] = x[:,0]*(self.params['i_max'] - self.params['i_min']) + self.params['i_min']
        x[:,1] = x[:,1]*(self.params['l_log_max'] - self.params['l_log_min']) + self.params['l_log_min']
        x[:,1] = 10**x[:,1]
        x[:,2] = x[:,2]*(self.params['a_max'] - self.params['a_min']) + self.params['a_min']
        return x
        
    def get_params(self):
        return self.params
    
    def save(self, filename):
        joblib.dump(self.params, filename)
        
    def load(self, filename):
        self.params = joblib.load(filename)
        
    
if __name__ == "__main__":
    # Test the transformer
    x = np.random.rand(10,3)
    transformer = Transformer()
    transformer.fit(x)
    print(transformer.get_params())
    print(x)
    transformer.transform(x)
    print(x)
    transformer.reverse_transform(x)
    print(x)
    
    transformer.save("transformer.pkl")
    transformer2 = Transformer()
    transformer2.load("transformer.pkl")
    print(transformer2.get_params())
    
    
    