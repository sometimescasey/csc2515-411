# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017
Revised by Casey Juanxi Li

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import math
from scipy.misc import logsumexp
import scipy

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

def calculate_A(test_datum, x_train, tau):
  test_datum2 = test_datum.reshape(test_datum.shape[0], 1).T

  # Nx1 array of squared norms
  norms = l2(x_train, test_datum2)
  exp_norms = np.exp(-1 * norms/(2*tau**2))
  b = np.amax(exp_norms)
  exp_norms_b = exp_norms - b

  
  print(norms.shape)
  denominator_b = np.sum(exp_norms_b, axis=0)

  N_x_train = x_train.shape[0]

  # Create A - N_x_train x N_x_train matrix of zeros
  A = np.zeros((N_x_train,N_x_train))

  # Calculate and set diagonals of A
  for i in range(0, N_x_train-1): 
    A[i][i] = exp_norms_b[i] / denominator_b 
  
  return A

# def bad_calculate_A(test_datum, x_train, tau):
#   test_datum2 = test_datum.reshape(test_datum.shape[0], 1).T

#   norms = l2(x_train, test_datum2)

#   N_x_train = x_train.shape[0]

#   # array of length N to store each exp(-|x - x(j)^2/2tau^2) element, 
#   # so we only have to calculate it once for each example from j = 0 to N
#   q = []
#   denominator = 0
  
#   # Populate q and calculate the denominator for each a_ii
#   for j in range(0, N_x_train-1):
#     q.append(math.exp(-1 * calc_norm(test_datum, x_train[j]) / (2 * tau**2)))
#     denominator += q[j]

#   # Create A - N_x_train x N_x_train matrix of zeros
#   A = np.zeros((N_x_train,N_x_train))

#   # Calculate and set diagonals of A
#   for i in range(0, N_x_train-1): 
#     A[i][i] = q[i] / denominator 
  
#   return A
 
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    A = calculate_A(test_datum, x_train, tau)
    N_x_train = x_train.shape[0]


    return None
    ## TODO




def run_validation(x,y,taus,frac_val):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO

    return None
    ## TODO

def main():
  # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    print(x.shape[0])
    print(x[0])
    print(x[1])

    taus = np.logspace(1.0,3,200)
    # print(taus)
    
    A = calculate_A(x[0], x, taus[0])
    print(A)

    train_losses, test_losses = run_validation(x,y,taus,frac_val=0.3)
    
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)


if __name__ == "__main__":
    main()


