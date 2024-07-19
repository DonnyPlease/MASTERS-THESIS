import numpy as np

import numpy as np
from scipy.optimize import fsolve

def cubic_roots(a, b, c, d):
    # Define the cubic equation
    def equation(x):
        return a * x**3 + b * x**2 + c * x + d
    
    # Initial guesses for the roots
    initial_guesses = np.array([0, 1, -1])
    
    # Find the roots using fsolve
    roots = fsolve(equation, initial_guesses)
    
    return roots

def numerical_partial_derivative(func, x, args, epsilon=1e-6):
    partial_derivatives = np.zeros(len(x))
    
    for i in range(len(x)):
        x1 = np.array(x, dtype=float)
        x2 = np.array(x, dtype=float)
        
        x1[i] -= epsilon
        x2[i] += epsilon
        
        partial_derivatives[i] = (func(x2, *args) - func(x1, *args)) / (2 * epsilon)
    
    return partial_derivatives

def propagate_error(a, b, c, d, sigma_a, sigma_b, sigma_c, sigma_d):
    coeffs = [a, b, c, d]
    sigmas = [sigma_a, sigma_b, sigma_c, sigma_d]
    
    # Calculate the roots at the original coefficients
    roots = cubic_roots(a, b, c, d)
    
    # Calculate the partial derivatives numerically
    partials = [numerical_partial_derivative(lambda coeffs, x: cubic_roots(*coeffs)[x], coeffs, (), epsilon=1e-6) for x in range(3)]
    
    # Propagate the errors
    errors = np.zeros(3)
    for i in range(3):
        errors[i] = np.sqrt(sum((partials[i][j] * sigmas[j])**2 for j in range(4)))
    
    return roots, errors

# Coefficients and their standard errors
a, b, c, d = 1, -6, 11, -6
sigma_a, sigma_b, sigma_c, sigma_d = 0.1, 0.1, 0.1, 0.1

# Propagate error
roots, errors = propagate_error(a, b, c, d, sigma_a, sigma_b, sigma_c, sigma_d)

print(f"The roots of the cubic equation are: {roots}")
print(f"The propagated errors are: {errors}")

        