
from __future__ import division
import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

# Select the number of labels
K = 6
#K = 6 # Set this to check results against the ones from available dataset

Y = np.arange(K) # Class set: array containing all the possible labels


def MLE(X_train, y_train, Y, K):
  """ This function calculates the maximum likelihood estimators of the probability vector pi, and of the parameters 
  of the class-conditional Gaussian distribution of the data.

  It takes the training datasets (covariates X_train and corresponding labels y_train), the set containing all possible
  labels Y, and the parameter K defined on top of the script.
  """
  _, d = X_train.shape # Dimension of the covariates
  N = len(y_train) # Number of samples in the training set
  n_y = np.zeros((K,1))
  pi_hat = np.zeros((K,1))
  mu_hat = np.zeros((d,K)) # K vectors arranged per columns in a matrix 
  Sigma_hat = np.zeros((d,d,K)) # K matrices
  for l in Y:
    for j in range(N):
      if y_train[j] == Y[l]:
        n_y[l] = n_y[l] + 1
        mu_hat[:,l] = mu_hat[:,l] + X_train[j]
    if n_y[l] < d: raise Exception('There are no sufficient observations of label %d' % l) # Note: investigate how to implement the classifier when this is not satisfied 
    pi_hat[l] = 1.0 / N * n_y[l]
    mu_hat[:,l] = 1.0 / n_y[l] * mu_hat[:,l] #Else, we leave it as zero? 
    for j in range(N):
      if y_train[j] == Y[l]:
        v = X_train[j] - mu_hat[:,l]
        Sigma_hat[:,:,l] = Sigma_hat[:,:,l] + np.outer(v, v)
    Sigma_hat[:,:,l] = 1.0 / n_y[l] * Sigma_hat[:,:,l] #Else, we leave the matrix entry with the zero? 
  # The next final lines check for positive definitiveness of the class-specific covariance matrices
  ev_product = np.ones(K) #Initialization of a variable whose entries will each contain the product of the eigenvalues of each of the class-specific covariance matrices
  ev = np.empty((d,K), dtype=float) #Initialization of a variable to store the eigenvalues of each of the class-specific covariance matrices
  for l in Y:
    ev[:,l], _ = np.linalg.eig(Sigma_hat[:,:,l])
    for j in range(d):
      ev_product[l] = ev_product[l] * ev[j,l] #The product of the (j+1)-th first eigenvalues of Sigma_hat. Eventually its value will be the full product of all the eigenvalues.
      if ev_product[l] < 0:
        raise Exception('There is at least a MLE of the covariance matrices that is not positive definite. Please, revise the code for errors.')
  return pi_hat, mu_hat, Sigma_hat


def pluginClassifier(X_train, y_train, X_test, Y, K, pi_hat, mu_hat, Sigma_hat):    
  """ This function calculates the predicted labels of the provided covariates X_test using the learned MLE parameters.
  It also outputs the probability of each of the possible labels for each provided covariate (i.e. the posterior of y).

  It takes:
  The training datasets (covariates X_train and corresponding labels y_train);
  The covariates to predict their labels X_test;
  The set containing all possible labels Y;
  The parameter K defined on top of the script;
  The MLE estimations of y's prior and of x's class-conditional distribution.
  """
  N_0, _ = X_test.shape # Number of vectors in the test set for which we are going to estimate their corresponding label 
  y_pred = np.empty(N_0) # Array that will store the predicted labels for the test data
  prior_of_x = np.zeros((N_0,1))
  posterior_of_y = np.zeros((N_0,K))
  for i in range(N_0): # The i-th iteration will find the prediction of the i-th label
    y_pred[i] = Y[0] # Let us initially predict it as the first label, to then update it as necessary
    exponent = -0.5 * np.matrix(X_test[i]-mu_hat[:,0]) * np.matrix(Sigma_hat[:,:,0]).I * np.matrix(X_test[i]-mu_hat[:,0]).T # Note the matrix objects are 1D arrays and are by default cast as row vectors (when using np.matrix(·))!
    magnitude_original =  np.linalg.det(Sigma_hat[:,:,0])**(-0.5) * np.exp(exponent) * pi_hat[0]
    prior_of_x[i] = prior_of_x[i] + magnitude_original # We add the first term of the cumulative sum
    posterior_of_y[i,0] = magnitude_original # This will eventually have the posterior probabilty of each of the labels 
    for l in range(K)[1:K]: # Note we extract the first index to avoid comparing the first label with itself
      exponent = -0.5 * np.matrix(X_test[i]-mu_hat[:,l]) * np.matrix(Sigma_hat[:,:,l]).I * np.matrix(X_test[i]-mu_hat[:,l]).T # Note the matrix objects are 1D arrays and are by default cast as row vectors (when using np.matrix(·))!
      magnitude_updated = np.linalg.det(Sigma_hat[:,:,l])**(-0.5) * np.exp(exponent) * pi_hat[l]
      prior_of_x[i] = prior_of_x[i] + magnitude_updated # We successively add the rest of the terms of the cumulative sum
      posterior_of_y[i,l] = posterior_of_y[i,l] + magnitude_updated # Successive computation of the sum in the numerator of the expression for p(y_i|x_i). (Partial result)
      if magnitude_updated > magnitude_original:
        y_pred[i] = Y[l] # This label gives a higher value of the expression inside the max function than the one assigned previously
        magnitude_original = magnitude_updated # The value inside the max function that the rest of candidate labels (that potentially may succeed the provisionally-assigned one) needs to be compared against that from the freshly-assigned label 
    posterior_of_y[i,:] = posterior_of_y[i,:] / prior_of_x[i] # Division over the denominator in the expression for p(y_i|x_i). (Final result)
  return y_pred, posterior_of_y

 
pi_hat, mu_hat, Sigma_hat = MLE(X_train, y_train, Y, K)
y_pred, posterior_of_y = pluginClassifier(X_train, y_train, X_test, Y, K, pi_hat, mu_hat, Sigma_hat)
np.savetxt("probs_test.csv", posterior_of_y, delimiter=",") # write output to file