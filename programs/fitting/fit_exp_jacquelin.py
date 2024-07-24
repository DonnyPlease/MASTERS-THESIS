import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import chisquare

class FitExp():
    """
    This class is handling fitting a sum of multiple exponential functions. 
    The whole fitting process is based on a (document) paper from Jean 
    Jaquelin. It works quite well with exponential decay and produces the best 
    results when the additive constant is kept - which is the usual practice in
    statistics even if we do not have an exact explanation of what the consant
    represents.
    """
    
    def __init__(self, x, y, exp_count=2, include_constant=False):
        self.x = x  
        self.y = y  
        self.data_size = len(x) 
        self.check_same_sizes(x, y)
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
                height = previous_integrals[i] + previous_integrals[i-1] 
                base = self.x[i]-self.x[i-1] 
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
        params = {}
        if self.include_constant:
            params['a0'] = a[0]
        for i in range(self.exponentials_count):
            params['a'+str(i+1)] = a[i+self.include_constant]
            params['b'+str(i+1)] = b[i]
        return params
    
    def _linearize_and_estimate_parameters(self):
        cumulative_integrals = self.get_cumulative_integrals()
        F = self.get_linearized_vector(cumulative_integrals)
        try:
            b_1 = np.dot(np.linalg.inv(np.matmul(F, np.transpose(F))), 
                        np.matmul(F, self.y)) 
        except:
            raise Exception("Could not calculate the inverse of the matrix 1") 
        return F, b_1
    
    def _estimate_errors(self, F, b_1):
        if self.exponentials_count == 2:
            dA, dB, *_ = self.std_error_of_fit_parameters(F, b_1)
            self.std_errors = self.std_error_of_b_coeffients_2exp(b_1[0],b_1[1], dA, dB)
        else:
            self.std_errors = None
    
    def _polynomial_roots(self, b_1):
        polynomial_coefs = self.get_polynomial_coefs(b_1)
        try:
            polynomial_roots = np.roots(polynomial_coefs)
        except:
            raise Exception("Could not calculate the roots of the polynomial") 
        # If polynomial_roots are complex, the fit is bad
        if any([np.imag(r) != 0 for r in polynomial_roots]):
            raise Exception("The roots of the polynomial are complex.")
        
        return polynomial_roots
    
    def _second_estimate_parameters(self, polynomial_roots):
        G = self.get_vector_of_exponentials(polynomial_roots)
        try:
            b_2 = np.dot(np.linalg.pinv(np.matmul(G, np.transpose(G))),
                                                    np.matmul(G, self.y))
        except:
            raise Exception("Could not calculate the inverse of the matrix 2")
        return G, b_2
    
    def _second_estimate_errors(self, G, b_2):
        if self.exponentials_count == 2:
            errors = self.std_error_of_lsq_parameters(G, self.y, b_2)
            if self.include_constant:
                self.std_errors = [errors[0], errors[1], self.std_errors[0], errors[2], self.std_errors[1]]
                return
            self.std_errors = [errors[0], self.std_errors[0], errors[1], self.std_errors[1]]
        
    
    def fit(self, verbose=False):
        # 1. LINEARIZATION AND EXPONENT PARAMETER ESTIMATION
        F, b_1 = self._linearize_and_estimate_parameters() 
        
        # 2. ERROR ESTIMATION
        self._estimate_errors(F, b_1)
            
        # 3. POLYNOMIAL COEFFICIENTS
        polynomial_roots = self._polynomial_roots(b_1)
        
        # 4. FINAL ESTIMATION OF PARAMETERS
        G,b_2 = self._second_estimate_parameters(polynomial_roots)
                              
        # 5. STANDARD ERROR OF FINAL COEFFICIENTS
        self._second_estimate_errors(G, b_2)
        
        # 6. CREATE A DICT OF PARAMETERS
        self.params = self.get_params(b_2, polynomial_roots)
        
        # 7. SHOW THE RESULTS (OPTIONAL)
        if verbose: 
            self.show_results()
        
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
    
    def std_error_of_lsq_parameters(self, X, y, b):
        residuals = y - np.dot(X.T, b)
        residual_sum_of_squares = np.sum(residuals**2)

        n = len(y)
        k = X.shape[0]
        variance_of_residuals = residual_sum_of_squares / (n - k)
        
        var_cov_matrix = np.linalg.inv(np.dot(X, X.T)) * variance_of_residuals
        variances = np.diag(var_cov_matrix)
        
        return np.sqrt(variances)
        
    def std_error_of_final_coeffients(self, A, B, dA, dB):
        # comes from error propagation
        D = B**2 + 4*A
        db1 = np.sqrt(dA**2/D + dB**2*(1/4)*(1+B/np.sqrt(D))**2)
        db2 = np.sqrt(dA**2/D + dB**2*(1/4)*(1-B/np.sqrt(D))**2)
        return db1, db2 
    
    def std_error_of_b_coeffients_2exp(self, A, B, dA, dB):
        # comes from error propagation
        D = B**2 + 4*A
        db1 = np.sqrt(dA**2/D + dB**2*(1/4)*(1+B/np.sqrt(D))**2)
        db2 = np.sqrt(dA**2/D + dB**2*(1/4)*(1-B/np.sqrt(D))**2)
        return db1, db2
    
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