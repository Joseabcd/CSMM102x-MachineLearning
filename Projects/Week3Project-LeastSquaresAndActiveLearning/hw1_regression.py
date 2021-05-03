import numpy as np
from numpy.linalg import inv

import sys


lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2]) # This is a parameter of the model for the likelihood distribution
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

m_X_train = np.matrix(X_train)
m_y_train = np.matrix(y_train).T # We treat vectors as column matrices
m_X_test = np.matrix(X_test)

_, d = X_train.shape

## Solution for Part 1
def part1(lambda_parameter, X, y, d):
    ## Input : Ridge Regression paremeter "lambda", the matrix "X" of measured data, the vector "y" of measured responses. 
    # Note that both matrices and vectors are expected as np.matrix objects.
    ## Return : the value of wRR
    parenthesis = lambda_parameter * np.matrix( np.eye(d,d) ) + X.T * X
    return parenthesis.I * X.T * y

wRR = part1(lambda_input, m_X_train, m_y_train, d)  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2(lambda_parameter, sigma2, X, X_new, d):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    n_0, _ = X_new.shape # Total number of measurements in the new set
    checking_list = list( range(n_0) ) # List of indexes for which we are going to go through to scan for maximum sigma2_0. (The indeces start at zero here.)
    indices_picked = np.zeros(10, dtype = int ) # Create an array with 10 elements (initialised to 0).
    update = np.matrix( np.zeros((d,d)) ) # Variable to update the covariance matrix of the w distribution when a new value is learnt.
    Identity = np.matrix( np.eye(d,d) )
    for r in range(10):
        Sigma = ( lambda_parameter * Identity + sigma2**(-2) * ( update + X.T * X ) ).I # Covariance matrix of w before updating with the new data
        i = checking_list[0] # The first data vector is checked outside the "for" loop below (in order to have an initial value for "maximum").
        maximum = sigma2 + X_new[i] * Sigma * X_new[i].T # Initial value for "maximum", to be updated below in the "for" loop. Please, bear in mind X_new[i] is (x_i)^T.
        indices_picked[r] = i # Index corresponding to the initial value of "maximum", to be updated below in the "for" loop when "maximum" be updated.
        for i in checking_list[1:]: # Note we've skipped the first case to check, for which the code has already gone through in the three previous lines.
            sigma2_0 = sigma2 + X_new[i] * Sigma * X_new[i].T
            if (sigma2_0 > maximum):
                maximum = sigma2_0
                indices_picked[r] = i # Index of the data picked in X_new, for which sigma2_0 will be the maximum
        sigma2 = np.asscalar(maximum) # We take its value as a scalar, since the result from the previous operations is a 1x1 matrix
        update = update + X_new[indices_picked[r]].T * X_new[indices_picked[r]]
        checking_list.remove(indices_picked[r]) # We remove the selected index from the list
    indices_picked = indices_picked + 1 # Index numbering will this way start at one, as required.
    return indices_picked 
        

active = part2(lambda_input, sigma2_input, m_X_train, m_X_test, d)  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active.reshape(1, active.shape[0]), delimiter=",", fmt='%i') # write output to file