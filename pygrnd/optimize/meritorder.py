import math as m
import numpy as np
import random
import itertools
from scipy.linalg import block_diag
from pygrnd.optimize.bruteforce import *


# QUBO knapsack with W as total size included as penalty

def QUBO_knapsack(values, weights, W, P):
    assert(len(values)==len(weights))
    M=np.zeros((len(values),len(values)))
    # -v_i on diagonal
    for i in range(len(values)):
        M[i][i]=values[i]
    # Constraints
    for i in range(len(values)):
        for j in range(len(values)):
            if i==j:
                M[i][j]=M[i][j]+P*((weights[i]**2-2*W*weights[i]))
            else:
                M[i][j]=M[i][j]+P*weights[i]*weights[j]
    return M
    
# params: List of values/weights
# b: number of bits in resolution

def splitParameters(params, b):
    res=[]
    for p in params:
        for j in range(b):
            res.append(p*(2**j)/(-1+2**b))
    return res

def recombineSolution(params, x, b):
    res=[]
    currentPos=0
    for i in range(len(params)//b):
        buffer=0.0
        for j in range(b):
            buffer=buffer+params[currentPos]*x[currentPos]
            currentPos=currentPos+1
        res.append(buffer)
    return res
    
    
# QUBO generation for upcost

def QUBO_switch(M, switchcost, uptime, resolution):

    Q_switch=np.zeros((len(M),len(M)))

    for x in range(0,len(uptime)):
        for i in range(0, resolution):
            for j in range(0,resolution):
                Q_switch[i+resolution*x][len(uptime)*uptime[x]*resolution+j+resolution*x]=-switchcost[x]
                Q_switch[len(uptime)*uptime[x]*resolution+j+resolution*x][i+resolution*x]=-switchcost[x]

    #print(Q_switch)
    return Q_switch



def QUBO_knapsack_slack_resolution(values, weights, W, b, P, verbose=False):
    values_copy=values.copy()
    weights_copy=weights.copy()
    values_copy.append(0)
    weights_copy.append(W)
    values_res=splitParameters(values_copy,b)
    weights_res=splitParameters(weights_copy,b)
    #print(values_res)
    M=QUBO_knapsack(values_res,weights_res,W,P)
    res_value,res_vector=solver(M)
    solutionVector=[]
    for i in range(np.shape(res_vector)[1]):
        solutionVector.append(res_vector[0,i])
    if verbose:
        print(solutionVector,res_value)
    values_sol=recombineSolution(values_res, solutionVector, b)
    weights_sol=recombineSolution(weights_res, solutionVector, b)
    fractions_helper=splitParameters([1]*len(values_copy),b)
    fractions_sol=recombineSolution(fractions_helper,solutionVector,b)
    slack_value=values_sol[-1]
    slack_weight=weights_sol[-1]
    if verbose:
        print("recombined fractions:",fractions_sol)
        print("recombined values:",values_sol)
        print("recombined weights:",weights_sol)
        print("total/slack value:",sum(values_sol)-slack_value,slack_value)
        print("real/slack weight:",sum(weights_sol)-slack_weight,slack_weight)
        print("total/demanded/diff weight:",sum(weights_sol),W,abs(sum(weights_sol)-W))
    return sum(values_sol)
