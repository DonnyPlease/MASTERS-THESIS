import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import chisquare

class FitExp():
    """
    This class is handling fitting a sum of multiple exponential functions. 
    The whole fitting process is based on this paper from Jean Jaquelin.
    It works quite well with exponential decay and produces the best results 
    when the additive constant is kept - which is the usual practice in
    statistic even if we do not have an exact explanation of what the consant
    represents.
    """
    
    def __init__(self, x, y, exp_count=2, constant=True):
        """The constructor of FitExp class.
        
        ----------
        parameters:
        ----------
        x : array-like - represents the x values of fitted data
        y : array-like - represents the y values of fitted data
        exp_count (optional) : int - number of exponential that are fitted 
        (default - 2)
        constant (optional) : bool - decides wether the constant is also fitted 
        or is excluded (default - True) 
        
        returns
        -------
        An instance of the class FitExp.
        """
        self.x = x  # Assign passed array to a member variable.
        self.y = y  # Assign passed array to a member variable.
        self.N = len(x)  # Assign length of the data to a member variable.
        if len(x) != len(y): 
            print("x and y have to be the same size or expect an exception.")
        self.exp_count = exp_count  # Assign the number of fitted exponentials 
                                    # to a member variable.
        self.constant = constant
        
    def _calculate_integrals(self):
        """(private functino) This function calculates the cumulative 
        intergrals by applying the trapezoidal formula.
        The number of integrals is specified when the class is instantiated.
        
        ----------
        parameters
        ----------
        none
        
        returns
        -------
        list of numpy arrays - each numpy array representing a cumulative 
        intergral and is of size self.N
        """
        integrals = []  # Initialize a variable.
        prev_int = self.y  # Initialize a variable so that the for loop does 
                           # not need condition for the first iteration.
        
        for i in range(self.exp_count):  # For each exponential function 
                                         # in the sum calculate the cumulative 
                                         # intergral from the previous 
                                         # integral.
            S = np.zeros(self.N)
            for i in range(1, self.N):
                a = prev_int[i] + prev_int[i-1]  # "bases" of the trapezoid
                b = self.x[i]-self.x[i-1]  # "height" of the trapezoid
                S[i] = S[i-1] + 1/2*a*b  # cumulation
            integrals.append(S)  # Append to the list.
            prev_int = S  # Change the variable prev_int so that in the next 
                          # iteration the integral is caluclated from the last 
                          # one.
        return integrals  # Return the list
    
    def _calculate_linearized_vector(self, integrals):
        """This function is responsible for creating a vector that later plays 
        the role of the data vector that is fitted using ordinary least squares.
        
        parameters
        ----------
        integrals - list of numpy arrays: list of cumulative intergrals
        
        returns
        -------
        numpy array  
        """
        size_of_vector = self.exp_count*2  # initilize size of vector
        if self.constant: size_of_vector += 1  # add one to the size if 
                                               # constant is also fitted
        vector = np.zeros((size_of_vector, self.N))  # initialize vector
        
        # assign integrals to the beginning of the vector
        for i in range(self.exp_count):
            vector[i,:] = integrals[self.exp_count-i-1]
        # append powers of x to the end of the vector
        for i in range(self.exp_count, size_of_vector):
            vector[i,:] = np.power(self.x, size_of_vector-i-1)
        return vector
    
    def _get_polynomial_coefs(self, b):
        """This function creates a list of polynomial coefficients with the 
        first coefficient being a one so it can be . The length of this list 
        is dependant on the number of the exponential functions that are being 
        fitted.
        
        parameters
        ----------
        b - list of parameters from the integral version of linear regression
        
        returns
        -------
        list of coefficients of a polynomial
        """
        coefs = [1]
        for i in range(self.exp_count):
            coefs.append(-b[self.exp_count-i-1])
        return coefs
            
    def _calculate_vector_of_exponentials(self,roots):
        """This function calculates a numpy array that represents the vector
        of exponentials from the paper.
        
        parameters
        ----------
        roots - roots of the polynomial calculated in the previous step of 
        the regression 
        
        returns
        -------
        numpy array repressenting 
        """
        size = self.exp_count
        if self.constant: size += 1
        vector = np.zeros((size, self.N))
        if self.constant: 
            vector[0,:] = np.ones(self.N)
            for i in range(self.exp_count):
                vector[i+1,:] = np.exp(roots[i]*self.x)
        else:
            for i in range(self.exp_count):
                vector[i,:] = np.exp(roots[i]*self.x)
        return vector

    def _get_params(self, roots, b_2):
        params = []
        if self.constant: 
            params.append(b_2[0])
            for i in range(self.exp_count):
                params.append(b_2[i+1])
                params.append(roots[i])
            return params

        for i in range(self.exp_count):
            params.append(b_2[i])
            params.append(roots[i])
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
        if self.constant: 
            fun += ' a0 + '
        for i in range(self.exp_count):
            fun += 'a{}.exp(b{}.x) + '.format(i+1,i+1)
        print(fun[:-2])
        print("-"*(18+len(fun)-3))
        print("Result of the fit gives us the following parameters:")
        c=0
        if self.constant: 
            print('     a0 = {:5f}'.format(self.params[0]))
            c=1
        for i in range(self.exp_count):
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
        res = self.y - y_fit
        print("     Mean of residuals = ", np.mean(res))
        print("     RMSE = ",np.sqrt(np.mean(np.power(res, 2))))
        
        print("#"*50)
        print("#"*50)
                
    
    def fit(self, verbose=False):
        """Function that performs the fit of sum of exponentials as it was 
        desrcibed in the article of Mr. Jean Jaqcuelin.
        
        parameters
        ---------------------------
        verbose - bool (default=False) - if True, there are results 
                                         printed in the console
        
        returns
        ---------------------------
        array of parameters of the exponential function
        
        """
        integrals = self._calculate_integrals()
        F = self._calculate_linearized_vector(integrals)
        b_1 = np.dot(np.linalg.inv(np.matmul(F, np.transpose(F))), 
                                                    np.matmul(F, self.y))    
        pol_coefs = self._get_polynomial_coefs(b_1)
        roots = np.roots(pol_coefs)
        G = self._calculate_vector_of_exponentials(roots)
        b_2 = np.dot(np.linalg.inv(np.matmul(G, np.transpose(G))),
                                                    np.matmul(G, self.y))
        self.params = self._get_params(roots, b_2)
        
        if verbose: self.show_results()

        return self.params
        
    def predict(self, x=None):
        """Function that calculates the prediction of the fitted function
        based on the fitted parameters.

        Args:
            x (np.array, optional): x values. Defaults to None. If None, the
            function uses the x values that were used to fit the function.

        Returns:
            np.array: prediction of the fitted function for the given x values.
        """
        if x == None:  # if x is not given, use the x values that were used
            x = self.x  # to fit the function

        prediction = np.zeros(len(x))  # initialize prediction
        
        # calculate the prediction
        if self.constant:
            prediction += self.params[0]
            for i in range(self.exp_count):
                prediction += self.params[i*2+1]*np.exp(self.params[i*2+2]*x)
        else:
            for i in range(self.exp_count):
                prediction += self.params[i*2]*np.exp(self.params[i*2+1]*x)
            
        return prediction
        
    def plot(self, log=False, residuals=False, show=True, save=False, 
             save_name='fit_result.pdf') -> None:
        """Function to plot the fitted function along with the data.

        parameters
        ----------
            log (bool, optional): Logarithmic scale. Defaults to False.
            residuals (bool, optional): If true, the plot also shows the 
                residuals. Defaults to False.
            show (bool, optional): If true, plot is shown during the run of 
                the program. Defaults to True.
            save (bool, optional): If true, the plot is saved to the location 
                given in a keyword parameter 'save_name'. Defaults to False.
            save_name (string, optional): Name of the file. Defaults to 
                'fit_result.pdf'.
            
        returns
        -------
        None
        """
        y_fit = self.predict()  # calculate the prediction
        res = self.y - y_fit  # calculate the residuals
        
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
        ax1.set_title('{}-exponential fit'.format(self.exp_count))
        
        # set the scale of the y axis
        if log: 
            ax1.set_yscale('log')
            ax1.set_title('''{}-exponential fit 
                          (log scale)'''.format(self.exp_count))
        
        # plot the residuals if the keyword parameter is set to True
        if residuals:
            ax2.scatter(self.x, res, c='black', label='Residuals')
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_title('Residuals')
            # ax2.set_ylim(-10,10)
        
        plt.tight_layout()  # make sure the labels are not cut off
        
        # show or save the plot
        if show:
            plt.show()
        if save:
            plt.savefig(save_name)
        
        # close the plot to avoid memory leaks
        plt.close()

        
        
        