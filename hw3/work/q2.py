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
from sklearn.model_selection import ShuffleSplit
import time
from sklearn.metrics import mean_squared_error

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
  scaled_norms = -1 * norms/(2*(tau**2))
  b = np.amax(scaled_norms) # for stability
  stable_exp = np.exp(scaled_norms - b)

  bottom = np.sum(stable_exp, axis=0)

  N_x_train = x_train.shape[0]

  # Create A - N_x_train x N_x_train matrix of zeros
  A = np.zeros((N_x_train,N_x_train))

  # Calculate and set diagonals of A
  for i in range(0, N_x_train): 
    A[i][i] = stable_exp[i] / bottom 
  
  return A

def calculate_A2(test_datum, x_train, tau):
  test_datum2 = test_datum.reshape(test_datum.shape[0], 1).T

  # Nx1 array of squared norms
  norms = l2(x_train, test_datum2)
  scaled_norms = -1 * norms/(2*(tau**2))
  b = np.amax(scaled_norms) # for stability
  
  top = np.exp(scaled_norms-b) # just in case I need the -b trick for the top as well
  bottom = np.exp(logsumexp(scaled_norms-b))

  N_x_train = x_train.shape[0]

  # Create A - N_x_train x N_x_train matrix of zeros
  A = np.zeros((N_x_train,N_x_train))

  # Calculate and set diagonals of A
  for i in range(0, N_x_train): 
    A[i][i] = top[i] / bottom 
  
  return A
 
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    A = calculate_A2(test_datum, x_train, tau)
    N_x_train = x_train.shape[0]
    D_x_train = x_train.shape[1]

    # rewrite (X.T * A * X - lambda*I)w = X.T * A *y as the system Pw = Q
    # define DxD matrix of 1's as I
    I = np.identity(D_x_train)
    X = x_train
    y = y_train

    P = np.matmul(np.matmul(X.T,A), X) - I*lam
    Q = np.matmul(np.matmul(X.T, A), y)

    best_w = solve_for_x(P, Q)

    y_pred_i = np.matmul(test_datum.T, best_w)

    return y_pred_i

def solve_for_x(A, b):
  # linearly solve for x in the system Ax = b
  x = np.linalg.solve(A,b)
  return x

def custom_train_test_split(X, y, test_size, random_state):
  # Borrowed from my HW1, since I'm unsure if we can use sklearn.model_selection.train_test_split 
  rs = ShuffleSplit(n_splits=1, test_size=test_size, 
    random_state=random_state)
  split = rs.split(X, y)
  
  for train_indices, test_indices in split:
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

  return X_train, X_test, y_train, y_test

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
    x_train, x_val, y_train, y_val = custom_train_test_split(x, y, frac_val, random_state=494)

    train_losses = []
    test_losses = []    
    
    for tau in taus:

      # not vectorized - how can this be done?
      print("tau: {}".format(tau))
      y_train_pred = []
      for i in range(0, x_train.shape[0]):
        y_train_pred.append(LRLS(x_train[i], x_train, y_train, tau, lam=1e-5))

      y_val_pred = []
      for i in range(0, x_val.shape[0]):
        y_val_pred.append(LRLS(x_val[i], x_train, y_train, tau, lam=1e-5))

      train_mse = 0.5 * mean_squared_error(y_train_pred, y_train)
      val_mse = 0.5 * mean_squared_error(y_val_pred, y_val)
      train_losses.append(train_mse)
      test_losses.append(val_mse) 
    
    return train_losses, test_losses


def main():
  # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish

    taus = np.logspace(1.0,3,200)

    start = time.time()
    train_losses, test_losses = run_validation(x,y,taus,frac_val=0.3)
    end = time.time()

    print("Done in {}".format(end - start))
    
    fig = plt.figure()
    plt.semilogx(taus, train_losses, label="Train losses", linewidth=1)
    plt.semilogx(taus, test_losses, label="Test losses", linewidth=1)
    fig.suptitle('Train and test error as function of tau', fontsize=20)
    plt.xlabel('tau', fontsize=18)
    plt.ylabel('average squared error loss', fontsize=16)
    plt.axis([10, 1000, -10, 100])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


