import numpy as np
#import pandas as pd
import scipy as sp
from scipy import stats
import sys


def KMeans(data):

	n,_ = X.shape #Number of data points
	mu = X[np.random.randint(0, high = n, size = K)].T #Initialization of the cluster centroids.
	c = np.zeros(n, dtype = int)
	for it in range(max_iter): #Number of iterations fixed to be max_iter
		for i in range(n):
			c[i] = np.argmin(np.linalg.norm(np.tile(X[i],(K,1)).T - mu, axis = 0)) #We calculate the norms of the differences between x_i and each of the centroids. Then pick the class for which that's minimum. 
		for k in range(K):
			mu[:,k] = np.sum(X[(c==k)], axis = 0) / np.sum((c==k)) #Note in the denominator that "True" counts as 1.
		filename = "centroids-" + str(it+1) + ".csv" 
		np.savetxt(filename, mu.T, delimiter=",")

def EMGMM(data):
    
	n,d = X.shape #Number of data points
	pi = np.full(K, 1.0 / K) #Initialization of the parameter for the prior on the cluster assignation.
	mu = X[np.random.randint(0, high = n, size = K)].T #Initialization of the K cluster-specific mean vectors.
	Sigma = np.zeros((K,d,d)); de = np.arange(d); Sigma[:,de,de] = 1.0 #Initialization of the K cluster-specific covariance matrices as identity matrices.
	phi = np.zeros((K,n))
	
	for it in range(max_iter): #Number of iterations fixed to be max_iter
		for i in range(n):
			den = np.sum(np.array([pi[j]*sp.stats.multivariate_normal.pdf(X[i],mu[:,j],Sigma[j]) for j in range(K)]))
			for k in range(K):
				phi[k,i] = pi[k] * sp.stats.multivariate_normal.pdf(X[i],mu[:,k],Sigma[k]) 
			phi[:,i] = phi[:,i] / den
		n_k = np.sum(phi, axis = 1) #A vector whose k-th entry contains n_k.
		pi = n_k / n #A vector whose k-th entry contains pi_k.
		mu = 1.0 / n_k * (phi @ X).T #See my notes to see how here the weighed sums are calculated.
		for k in range(K):
			Sigma[k] = 1.0 / n_k[k] * (phi[k] * (X - mu[:,k]).T @ (X - mu[:,k])) #Note transposition has been applied twice (though not explicitly indicated and performed)

		filename = "pi-" + str(it+1) + ".csv" 
		np.savetxt(filename, pi, delimiter=",") 
		filename = "mu-" + str(it+1) + ".csv"
		np.savetxt(filename, mu.T, delimiter=",") #this must be done at every iteration
		for k in range(K): #k is the number of clusters 
			filename = "Sigma-" + str(k+1) + "-" + str(it+1) + ".csv" #this must be done K times for each iteration
			np.savetxt(filename, Sigma[k], delimiter=",")

if __name__ == '__main__': #When this script be run, the __name__ variable will equal to __main__. When it be imported to another script, within this script's environment __name__ will be equal to the name of this script (and in that case the actions below are not performed; only the definitions outside the "if" above would be run).

	X = np.genfromtxt(sys.argv[1], delimiter = ",")
	K = 3 #Input the desired number of clusters
	max_iter = 10 #Unless we implement a stopping criterion, the number of iterations is always this number.

	data = [X,K,max_iter]
	
	KMeans(data)  
	EMGMM(data)
