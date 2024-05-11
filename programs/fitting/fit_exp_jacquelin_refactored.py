import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import chisquare

class FitExp():
    """
    This class is handling fitting a sum of multiple exponential functions. 
    The whole fitting process is based on this (document)paper from Jean Jaquelin.
    It works quite well with exponential decay and produces the best results 
    when the additive constant is kept - which is the usual practice in
    statistic even if we do not have an exact explanation of what the consant
    represents.
    """
    
    def __init__(self, x, y, exp_count=2, include_constant=False):
        self.x = x  
        self.y = y  
        self.data_size = len(x) 
        self.check_same_sizes(x, y)  # Check if the sizes of x and y are the same.
        self.exponentials_count = exp_count
        self.include_constant = include_constant
    
    def check_same_sizes(self, x, y):
        if len(x) != len(y):
            raise ValueError("arrays x and y have to be the same size")
    
    def get_cumulative_integrals(self):
        integrals = [] 
        previous_integrals = self.y 
        
        # Calculate cumulative (integrals) of y for linearization
        for i in range(self.exponentials_count):
            current_integrals = np.zeros(self.data_size)
            for i in range(1, self.data_size):
                # trapezoidal approximation of the integral
                base = previous_integrals[i] + previous_integrals[i-1] 
                height = self.x[i]-self.x[i-1] 
                current_integrals[i] = current_integrals[i-1] + 1/2*base*height 
            integrals.append(current_integrals)  
            previous_integrals = current_integrals 
        return integrals
    
    def get_linearized_vector(self, integrals):
        size_of_vector = self.exponentials_count*2 + self.include_constant 
        vector = np.zeros((size_of_vector, self.data_size))  # initialize vector
        
        # assign cumulative integrals to the beginning of the vector
        for i in range(self.exponentials_count):
            vector[i,:] = integrals[self.exponentials_count-i-1]
            
        # append powers of x to the end of the vector
        for i in range(self.exponentials_count, size_of_vector):
            vector[i,:] = np.power(self.x, size_of_vector-i-1)
        return vector
    
    def get_polynomial_coefs(self, b):
        coefs = [1]
        for i in range(self.exponentials_count):
            coefs.append(-b[self.exponentials_count-i-1])
        return coefs
            
    def get_vector_of_exponentials(self, parameters):
        # returns [[exp(b1*x1), exp(b1*x2), ...], [exp(b2*x1), ...], ...] where
        # parameters = [b1, b2, ...]
        # if the constant is included, the first element of the vector is 1
        vector_size = self.exponentials_count + self.include_constant
        vector = np.zeros((vector_size, self.data_size))
        
        if self.include_constant:
            vector[0,:] = np.ones(self.data_size)
            
        for i in range(self.exponentials_count):
            j = i + self.include_constant
            vector[j, :] = np.exp(parameters[i]*self.x)
            
        return vector

    def get_params(self, a, b):
        # returns [a0, a1, b1, a2, b2, ...] where a0 is the constant (0 if not included)
        params = []
        params.append(a[0]*self.include_constant)
        for i in range(self.exponentials_count):
            params.append(a[i])
            params.append(b[i])
        return params
    
    def show_results(self) -> None:
        """Print the results of the fit in the console.
        
        parameters
        ----------
        none
        
        returns
        -------
        None
        """
        print("#"*70)
        print()
        print("Fitted function:  ",end="")
        fun = "f(x) ="
        if self.include_constant: 
            fun += ' a0 + '
        for i in range(self.exponentials_count):
            fun += 'a{}.exp(b{}.x) + '.format(i+1,i+1)
        print(fun[:-2])
        print("-"*(18+len(fun)-3))
        print("Result of the fit gives us the following parameters:")
        c=0
        if self.include_constant: 
            print('     a0 = {:5f}'.format(self.params[0]))
            c=1
        for i in range(self.exponentials_count):
            print('     a{} = {:5f}'.format(i+1,self.params[i*2+c]))
            print('     b{} = {:5f}'.format(i+1, self.params[i*2+1+c]))
        
        print("Stats: ")
        try:
            chi2 = chisquare(self.y, self.predict())   
            print("     Chi2 = {:5f}".format(chi2[0]))    
            print("     Chi2 p_value = {:5f}".format(chi2[1]))
        except:
            print("Cannot calculate chi squared") 
        
        try:    
            print("     R-squared = ", metrics.r2_score(self.y,self.predict()))
        except:
            print("Cannot calculate R-squared statistic.")
            
        y_fit = self.predict()
        try:
            res = np.log(self.y+1) - np.log(y_fit+1)
            print("     Mean of residuals = ", np.mean(res))
            print("     RMSE = ",np.sqrt(np.mean(np.power(res, 2))))
        except:
            print("     Mean of residuals could not be calculated because of the logs")
            print("     It is possible that the predicted values are negative if the constant is negative")
        
        print("#"*50)
        print("#"*50)
    
    def fit(self, verbose=False):
        cumulative_integrals = self.get_cumulative_integrals()
        F = self.get_linearized_vector(cumulative_integrals)
        b_1 = np.dot(np.linalg.inv(np.matmul(F, np.transpose(F))), np.matmul(F, self.y))    
        
        self.std_errors = self.std_error_of_fit_parameters(F, b_1)
        self.std_errors = self.std_error_of_final_coeffients(b_1[0], b_1[1], self.std_errors[0], self.std_errors[1])
        polynomial_coefs = self.get_polynomial_coefs(b_1)
        polynomial_roots = np.roots(polynomial_coefs)
        G = self.get_vector_of_exponentials(polynomial_roots)
        b_2 = np.dot(np.linalg.inv(np.matmul(G, np.transpose(G))),
                                                    np.matmul(G, self.y))
        self.params = self.get_params(b_2, polynomial_roots)
        
        if verbose: self.show_results()

        return self.params
    
    def std_error_of_fit_parameters(self, X, b):
        residuals = self.y - np.dot(X.T, b)
        residual_sum_of_squares = np.sum(residuals**2)

        n = len(self.y)
        k = X.shape[0]  # Number of predictors (columns of F), including intercept
        variance_of_residuals = residual_sum_of_squares / (n - k - 1)

        # Variance-Covariance Matrix
        var_cov_matrix = np.linalg.inv(np.dot(X, X.T)) * variance_of_residuals
        variances = np.diag(var_cov_matrix)

        return np.sqrt(variances)
    
    def std_error_of_final_coeffients(self, A, B, dA, dB):
        db1 = np.sqrt(dA**2/(B**2+4*A)+dB**2*1/4*(1+B/np.sqrt(B**2+4*A))**2)
        db2 = np.sqrt(dA**2/(B**2+4*A)+dB**2*1/4*(1-B/np.sqrt(B**2+4*A))**2)
        return [0, 0, db1, 0, db2]
      
    def predict(self, x=None):
        if x is None:   # If x is not given, use the x values that were used
            x = self.x  # to fit the function.

        prediction = np.zeros(len(x)) 
        
        # calculate the prediction
        prediction += self.params[0]
        for i in range(self.exponentials_count):
            prediction += self.params[i*2+1]*np.exp(self.params[i*2+2]*x)
            
        return prediction
        
    def plot(self, log=False, residuals=False, show=True, save=False, 
             save_name='fit_result.pdf') -> None:
        
        y_fit = self.predict() 
        res = np.log(self.y+1) - np.log(y_fit+1)  # calculate the residuals TODO: residuals ->> log(residuals)
        
        # create the plot
        fig, (ax1,ax2) = plt.subplots(2,1,
                                      sharex=False, 
                                      gridspec_kw={'height_ratios': [4, 1]}, 
                                      figsize=(8,6)) 
        ax1.scatter(self.x, self.y)  # plot the data
        ax1.plot(self.x, y_fit, c='r')  # plot the fitted function
        
        # name the axes and the title
        ax1.set_xlabel('E [keV]')
        ax1.set_ylabel('Electron counts')
        
        # set the scale of the y axis
        if log: 
            ax1.set_yscale('log')
        
        # plot the residuals if the keyword parameter is set to True
        if residuals:
            ax2.scatter(self.x, res, c='black', label='Residuals')
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_title('Residuals')
            # ax2.set_ylim(-10,10)
        
        plt.tight_layout()  # make sure the labels are not cut off
        
        # show the plot if the keyword parameter is set to True
        
        if show:
            plt.show()
            
        # save the plot if the keyword parameter is set to True    
        if save:
            try:
                plt.savefig(save_name)
            except:
                print("Could not save the plot.")
        # close the plot to avoid memory leaks
        plt.close()

        
        
        