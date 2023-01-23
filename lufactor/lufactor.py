# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 09:32:29 2023

@author: rvaug
"""

import numpy as np

def lu_crout(A):
    n = np.shape(A)[0]
    Q = np.zeros_like(A)
    
    # first iteration unwrapped since it is just a direct copy of A
    Q[:,0] = A[:,0]
    Q[0,1:] = A[0,1:] / A[0,0]
    
    # python matrix multiplication raises error for multiplying 1x1 matrix
    # therefore second iteration also unwrapped
    Q[1:n,1] = A[1:n,1] - Q[1:n,0] * Q[0,1]
    Q[1,2:n] = ( A[1,2:n] - Q[1,0] * Q[0,2:n] ) / Q[1,1]
    
    for j in range(2,n-1): # unwrapped first, second, and last iteration
        i = range(j) # doesn't include end bound 0,1,2,...,j-1
        
        k = range(j,n)
        Q[k,j] = A[k,j] - np.dot(Q[k,:][:,i], Q[i,j])
        
        k = range(j+1,n)
        Q[j,k] = ( A[j,k] - np.dot(Q[j,i], Q[i,:][:,k]) ) / Q[j,j]
    
    # last iteration unwrapped because only a column, no row
    Q[n-1,n-1] = A[n-1,n-1] - np.dot(Q[n-1,:n-1], Q[:n-1,n-1])
        
    return Q