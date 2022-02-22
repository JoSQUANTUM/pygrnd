## brute force solver
## iterate through full list of possible solutions

import math as m
import numpy as np
import random
import itertools
from scipy.linalg import block_diag

def solver(M):
    n=len(M)
    minimum_value=0
    minimum_vector= np.matrix([[0]*n])
    tuples=itertools.product(*[(0, 1)]*n)
    for t in tuples:
        v=np.matrix(list(t))
        res=np.matmul(np.matmul(v,M),np.transpose(v))[0,0]
        if res<minimum_value:
            minimum_value=res
            minimum_vector=v
    return minimum_value, minimum_vector

# solving the QUBO with MC
# random vector multiplication

def MCsolver(M):
    n=len(M)
    minimum_value=0
    for t in range(1,1000000):
        v=np.random.randint(2, size=(1, n))
        res=np.matmul(np.matmul(v,M),np.transpose(v))[0,0]
        if res<minimum_value:
            minimum_value=res
            minimum_vector=v
    return minimum_value, minimum_vector
