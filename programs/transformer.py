import numpy as np
import joblib

def transform(x, y, factor_i = 1, factor_l = 1, factor_a = 1):
    t = Transformer()
    t.fit(x)
    x = t.transform(x, factor_i, factor_l, factor_a)
    y = np.array(y)
    return x, y, t

class Transformer:
    def __init__(self):
        self.x = None
        self.params = {}
        return
    
    def fit(self, x):
        self.x = x
        self.params['i_log_min'] = np.min(np.log10(x[:,0]))
        self.params['i_log_max'] = np.max(np.log10(x[:,0]))
        
        self.params['l_log_min'] = np.min(np.log10(x[:,1]))
        self.params['l_log_max'] = np.max(np.log10(x[:,1]))
        
        self.params['a_min'] = np.min(x[:,2])
        self.params['a_max'] = np.max(x[:,2])
        
    def transform(self, x, factor_i = 1, factor_l = 1, factor_a = 1):
        x[:,0] = self.transform_i(x[:,0], factor_i)
        x[:,1] = self.transform_l(x[:,1], factor_l)
        x[:,2] = self.transform_a(x[:,2], factor_a)
        return x
    
    def reverse_transform(self, x, factor_i = 1, factor_l = 1, factor_a = 1):
        x[:,0] = self.reverse_transform_i(x[:,0], factor_i)
        x[:,1] = self.reverse_transform_l(x[:,1], factor_l)
        x[:,2] = self.reverse_transform_a(x[:,2], factor_a)
        return x
        
    def get_params(self):
        return self.params
    
    def save(self, filename):
        joblib.dump(self.params, filename)
        
    def load(self, filename):
        self.params = joblib.load(filename)
        
    def transform_i(self, x, factor = 1):
        x = np.log10(x)
        x = (x - self.params['i_log_min'])/(self.params['i_log_max'] - self.params['i_log_min'])
        return x*factor

    def transform_l(self, x, factor = 1):
        x = np.log10(x)
        x = (x - self.params['l_log_min'])/(self.params['l_log_max'] - self.params['l_log_min'])
        return x*factor
    
    def transform_a(self, x, factor = 1):
        x = (x - self.params['a_min'])/(self.params['a_max'] - self.params['a_min'])
        return x*factor
    
    def reverse_transform_i(self, x, factor = 1):
        x /= factor
        x = x*(self.params['i_log_max'] - self.params['i_log_min']) + self.params['i_log_min']
        x = 10**x
        return x
    
    def reverse_transform_l(self, x, factor = 1):
        x /= factor
        x = x*(self.params['l_log_max'] - self.params['l_log_min']) + self.params['l_log_min']
        x = 10**x
        return x
    
    def reverse_transform_a(self, x, factor = 1):
        x = x/factor*(self.params['a_max'] - self.params['a_min']) + self.params['a_min']
        return x
        
    
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