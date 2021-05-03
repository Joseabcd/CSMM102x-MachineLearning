from __future__ import division
import numpy as np
import sys
from copy import deepcopy

train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
sigma2 = 0.1
d = 5

max_iter = 50

# Implement function here
def PMF(train_data,lam,sigma2,d):
    N1 = len(set(train_data[:,0]))
    N2 = len(set(train_data[:,1]))
    M = -1 * np.ones((N1,N2)) # Let's consider missing data are represented by a -1 at the corresponding matrix position (we're assuming there can be no rating equal to -1).
    Eye = np.eye((d))
    U = np.zeros((N1,d))
    V = np.zeros((N2,d))
    
    # We create some lists that will record the obtained vector matrices and loss at each iteration
    U_matrices = []
    V_matrices = []
    L_values = []
 
    # First we fill the matrix of ratings from the provided list: 
    for row in range(train_data.shape[0]):
        M[int(train_data[row,0])-1,int(train_data[row,1])-1] = train_data[row,2]
    
    # Next we compute iteratively (since we didn't have a closed-form solution) the ML solution for U and V.
    LamSigma2Eye = lam * sigma2 * Eye # We precompute this constant matrix that appears in the updates in order to save operations.
    V = np.genfromtxt(sys.argv[2], delimiter = ",")
    #V[:] = np.random.multivariate_normal(np.zeros(d), (lam**(-1.0))*Eye) # Initialization of the object vectors in order to have an initial value to substitute in the expression for the user vectors.
    # Update the vectors until the objective function's value doesn't change anymore (i.e. until we find the user and object vectors which satisfy the equations for both simultaneously).
    for _ in range(max_iter):
        for i in range(N1): 
            mask = (M[i,:] != -1)
            U[i] = np.linalg.inv( LamSigma2Eye + np.transpose(V[mask]).dot(V[mask]) ).dot( np.sum( M[i,mask] * np.transpose(V[mask]), axis=1 ) )
        for j in range(N2):
            mask = (M[:,j] != -1)
            V[j] = np.linalg.inv( LamSigma2Eye + np.transpose(U[mask]).dot(U[mask]) ).dot( np.sum( M[mask,j] * np.transpose(U[mask]), axis=1 )) # note M[mask,j] is a 1D array and is therefore taken as a (1 x length) rank-2 array in the broadcasting process.
        mask2D = (M != -1) # Elements of the matrix that contain ratings
        prediction = lambda pair:  np.inner(U[pair[0]],V[pair[1]]) # A function to calculate only the needed predictions (i.e. the ones for which we already have rating)
        pairs = zip(np.where(mask2D)[0],np.where(mask2D)[1]) # The needed predictions are those for which we already have rating. We create a list of pairs of user-object for which we have rating.
        L_values += [ -1.0/(2.0*sigma2) * ( ( M[mask2D] - list(map(prediction, pairs)) )**2 ).sum() - lam/2.0 * ( (U**2).sum() + (V**2).sum() ) ]
        deep_copied_array = deepcopy(U) # If we don't do (something like) this then the previously concatenated lists are modified in the assignments above!!!!!!
        U_matrices += [deep_copied_array]
        deep_copied_array = deepcopy(V)
        V_matrices += [deep_copied_array]

    return L_values, U_matrices, V_matrices


# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L_values, U_matrices, V_matrices = PMF(train_data,lam,sigma2,d)

np.savetxt("objective.csv", L_values, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")