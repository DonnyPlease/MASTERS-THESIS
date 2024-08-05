import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

from scipy.optimize import minimize as scipy_minimize

def plot_histogram_with_double_exp(x, y, params):
    plt.scatter(x, y)
    y_double_exp = double_exp(x, params)
    plt.plot(x, y_double_exp, color='red')
    plt.yscale('log')
    plt.show()
    
def double_exp(x,params):
    a0, a1, b1, a2, b2 = params
    return a0 + a1*np.exp(-x*b1) + a2*np.exp(-x*b2)

def weighted_squared_error(y, y_pred):
    return np.mean(((y-y_pred)**2)/y)

def one_double_exp_fit(x, y, b1, b2):
    X = np.stack([np.ones(x.shape), np.exp(-x*b1), np.exp(-x*b2)], axis=-1)
    b = np.linalg.inv(X.T@X)@X.T@y
    params = [b[0], b[1], b1, b[2], b2]
    return params

def weighted_squared_error_of_linear_regression(x, y, b1, b2):
    X = np.stack([np.exp(-x*b1), np.exp(-x*b2)], axis=-1)
    reg = LinearRegression().fit(X, y)
    coefs = [reg.intercept_, reg.coef_[0], b1, reg.coef_[1], b2]
    return weighted_squared_error(y, double_exp(x, coefs))
    

def scan_fit_double_exp(x, y):
    parameters_range = np.logspace(-6, 1, 100)
    
    errors = np.ones((len(parameters_range), len(parameters_range)))*1e16
    for i in range(1,len(parameters_range)):
        for j in range(i):
            b1 = parameters_range[i]
            b2 = parameters_range[j]
            # print(b1, b2)
            X = np.stack([np.exp(-x*b1), np.exp(-x*b2)], axis=-1)
            # params = one_double_exp_fit(x, y, parameters_range[i], parameters_range[j])
            reg = LinearRegression().fit(X, y)
            coefs = [reg.intercept_, reg.coef_[0], b1, reg.coef_[1], b2]
            errors[i,j] = weighted_squared_error(y, double_exp(x, coefs))
            
    plt.imshow(errors)
    plt.show()
    
    # Find the best parameters
    min_error = np.min(errors)
    print(min_error)
    best_params = np.where(errors == min_error)
    b1 = parameters_range[best_params[0][0]]
    b2 = parameters_range[best_params[1][0]]
    print(b1, b2)
    return b1, b2
    # # Fit the double exponential
    # X = np.stack([np.exp(-x*b1), np.exp(-x*b2)], axis=-1)
    # reg = LinearRegression().fit(X, y)
    # coefs = [reg.intercept_, reg.coef_[0], b1, reg.coef_[1], b2]
    # print(coefs)
    # plot_histogram_with_double_exp(x, y, coefs)

def variances_of_parameters(x, params, s_squared_estimate):
    a0, a1, b1, a2, b2 = params
    jacobian = np.array([[1 for _ in range(len(x))],
                         np.exp(-x*b1),
                         [-a1*x_i*np.exp(-x_i*b1) for x_i in x],
                         np.exp(-x*b2),
                         [-a2*x_i*np.exp(-x_i*b2) for x_i in x]])
    cov_matrix = np.linalg.inv(jacobian@jacobian.T)
    return np.sqrt(np.diagonal(cov_matrix))*s_squared_estimate


class ParameterOptimizer():
    def __init__(self, x, y):
        self.x = x[40:-20]
        self.y = y[40:-20]
        self.params = None
        
    def error_func(self, params):
        b1, b2 = params
        return weighted_squared_error_of_linear_regression(self.x, self.y, b1, b2)
    
    def optimize_double_exp(self, initial_guess=[0.094, 0.025]):
        # Run the optimization
        result = scipy_minimize(self.error_func, initial_guess, method='nelder-mead')
        
        # Extract the optimized parameters
        b1_opt, b2_opt = result.x
        
        print("Optimized parameters: b1 =", b1_opt, ", b2 =", b2_opt)
        
        params = one_double_exp_fit(self.x, self.y, b1_opt, b2_opt)
        self.params = params
        plot_histogram_with_double_exp(self.x, self.y, params)
    
    def get_temperatures(self):
        a0, a1, b1, a2, b2 = self.params
        T1 = 1/b1
        T2 = 1/b2
        return T1, T2
    
    def print_params(self):
        print("Params: ", self.params)
    
    def print_temperatures(self):
        print("T1 =", 1/self.params[2], "T2 =", 1/self.params[4])
        
    def get_y_pred(self, x):
        return double_exp(x, self.params)
    
    def get_residuals(self):
        return self.y - self.get_y_pred(self.x)
    
    def get_mean_weighted_squared_error(self):
        return weighted_squared_error(self.y, self.get_y_pred(self.x))
    
    def get_variance_estimate(self):
        return self.get_mean_weighted_squared_error()
    
    def get_variances_of_parameters(self):
        return variances_of_parameters(self.x, self.params, self.get_variance_estimate())