from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.basic_provider import BasicProvider
from qiskit import transpile
from qiskit.visualization import plot_histogram

import itertools
import math
from math import pi
import cmath
import random
import numpy as np

from scipy.optimize import minimize

import matplotlib.pyplot as plt
from scipy.stats import norm


def num2bin(x,r):
    res=""
    buffer=x
    for i in range(r):
        m=buffer%2
        res=res+str(m)
        buffer=(buffer-m)//2
    return res[::-1]

def allCombinations(n):
    if n==0:
        return ['']
    res=[]
    for a in allCombinations(n-1):
        res.append(a+'0')
        res.append(a+'1')
    return res
#allCombinations(3)

def counts2probs(counts):
    bits=len(list(counts.keys())[0])
    summe=0
    for a in allCombinations(bits):
        if a in counts:
            summe=summe+counts[a]
    res=[]
    for a in allCombinations(bits):
        if a in counts:
            res.append(counts[a]/summe)
        else:
            res.append(0.0)
    return res

def fidelityCounts(countsP, countsQ):
    prob0=counts2probs(countsP)
    prob1=counts2probs(countsQ)
    summe=0
    for i in range(len(prob0)):
        summe=summe+math.sqrt(prob0[i]*prob1[i])
    return summe**2



def maxString(counts):
    bits=len(list(counts.keys())[0])
    probList=counts2probs(counts)
    stringList=allCombinations(bits)
    maxi=np.argmax(probList)
    return stringList[maxi]

# Return objective to x^t Q x value
def eval_solution(x,m):
    obj=np.matmul(np.matmul(x,m),np.transpose(x))
    return obj



# Brute force QUBO solver
def bruteForceSolver(m):
    menge=[p for p in itertools.product([0,1], repeat=len(m))]
    #bestCost=-math.inf # maximize
    bestCost=math.inf # minimize
    bestVector=0
    for x in menge:
        v=np.zeros((1,len(m)))
        for i in range(len(m)):
            v[0,i]=x[i]
        value=np.matmul(v,np.matmul(m,np.transpose(v)))
        if value<bestCost: #value>bestCost: <-- maximize
            bestCost=value
            bestVector=x
    return bestVector,bestCost


# Convert matrix
def matrixConvertInv(m):
    buffer=np.zeros((len(m),len(m)))
    for i in range(len(m)):
        for j in range(len(m)):
            if i==j:
                buffer[i][i]=buffer[i][i]+m[i][i]/2
            else:
                buffer[i][i]=buffer[i][i]+m[i][j]/4
                buffer[j][j]=buffer[j][j]+m[i][j]/4
                buffer[i][j]=buffer[i][j]+m[i][j]/4
    return buffer

# Cost unitary for QAOA implementation
def addGates(qr,qc,m1, gamma):
    for i in range(len(m1)):
        for j in range(len(m1)):
            if i==j:
                qc.rz(gamma*m1[i][i],qr[i])
                #qc.barrier()
            elif abs(m1[i][j])>0.00001:
                qc.cx(qr[i],qr[j])
                qc.rz(gamma*m1[i][j],qr[j])
                qc.cx(qr[i],qr[j])
                #qc.barrier()
    #qc.barrier()
    return qc


## Single layer QAOA

def qaoaExp(m0,beta,gamma,Nshots,backend = BasicProvider().get_backend("basic_simulator")):
    
    m=matrixConvertInv(m0)
    
    qr=QuantumRegister(len(m),'q')
    cr=ClassicalRegister(len(m),'c')
    qc=QuantumCircuit(qr,cr)
    for i in range(len(m)):
        qc.h(qr[i])
    addGates(qr,qc,m,gamma)
    for i in range(len(m)):
        qc.ry(beta,qr[i])
    qc.measure(qr,cr)

    #backend=Aer.get_backend("qasm_simulator")
    #job = execute(qc, backend,shots=Nshots)
    #counts=job.result().get_counts()

    #backend = BasicProvider().get_backend("basic_simulator")
    
    qcNew=transpile(qc, backend)
    job=backend.run(qcNew,shots=Nshots)
    counts=job.result().get_counts()

    # pull result with highest count
    maxValue=-math.inf
    for c in counts:
        if counts[c]>maxValue:
            maxValue=counts[c]
            prob=maxValue/Nshots
    for c in counts:
        if counts[c]==maxValue:
            #print(c,counts[c])
            vec=c

    # calculate cost
    x=[]
    for i in maxString(counts):
        x.append(int(i))

    obj=eval_solution(x,m0)

    return vec, counts, obj, qc, prob




##### Optimize multiLayerqaoaExp with different numbers of parameters
#- use minimize with a function that has a single parameter (can be a vector)
#- define appropriate function based on multyLayerqaoaExp
#- use first half as $\beta$ parameters and second half as $\gamma$ parameters
#- try one, two or six layers (2, 4 or 12 total parameters)

# calculate expectation value through multi layer qaoa

def multiLayerqaoaExp(m,betas,gammas,Nshots,backend = BasicProvider().get_backend("basic_simulator")):
    
    mPauli=matrixConvertInv(m)
    
    qr=QuantumRegister(len(mPauli),'q')
    cr=ClassicalRegister(len(mPauli),'c')
    qc=QuantumCircuit(qr,cr)

    
    for i in range(len(mPauli)):
        qc.h(qr[i])
    qc.barrier()
        
    for j in range(len(betas)):
        
        addGates(qr,qc,mPauli,gammas[j])
        qc.barrier()
        for k in range(len(mPauli)):
            qc.ry(betas[j],qr[k])
            #qc.barrier()
        qc.barrier()
    qc.measure(qr,cr)

    #backend=Aer.get_backend("qasm_simulator")
    #job = execute(qc, backend,shots=Nshots)
    #counts=job.result().get_counts()

    qcNew=transpile(qc, backend)
    job=backend.run(qcNew,shots=Nshots)
    counts=job.result().get_counts()

    # pull result with highest count
    maxValue=-math.inf
    for c in counts:
        if counts[c]>maxValue:
            maxValue=counts[c]
            prob=maxValue/Nshots
    for c in counts:
        if counts[c]==maxValue:
            #print(c,counts[c])
            vec=c

    # calculate cost
    x=[]
    for i in maxString(counts):
        x.append(int(i))

    obj=eval_solution(x,m)

    #return vec, sol, counts, obj, prob, qc
    return obj


# calculate expectation value through multi layer qaoa

def multiLayerqaoa(m,betas,gammas,Nshots,backend = BasicProvider().get_backend("basic_simulator")):
    
    mPauli=matrixConvertInv(m)
    
    qr=QuantumRegister(len(mPauli),'q')
    cr=ClassicalRegister(len(mPauli),'c')
    qc=QuantumCircuit(qr,cr)

    
    for i in range(len(mPauli)):
        qc.h(qr[i])
    qc.barrier()
        
    for j in range(len(betas)):
        
        addGates(qr,qc,mPauli,gammas[j])
        qc.barrier()
        for k in range(len(mPauli)):
            qc.ry(betas[j],qr[k])
            #qc.barrier()
        qc.barrier()
    qc.measure(qr,cr)

    #backend=Aer.get_backend("qasm_simulator")
    #job = execute(qc, backend,shots=Nshots)
    #counts=job.result().get_counts()

    qcNew=transpile(qc, backend)
    job=backend.run(qcNew,shots=Nshots)
    counts=job.result().get_counts()

    # pull result with highest count
    maxValue=-math.inf
    for c in counts:
        if counts[c]>maxValue:
            maxValue=counts[c]
            prob=maxValue/Nshots
    for c in counts:
        if counts[c]==maxValue:
            #print(c,counts[c])
            vec=c

    # calculate cost
    x=[]
    for i in maxString(counts):
        x.append(int(i))

    obj=eval_solution(x,m)

    return vec, counts, obj, prob, qc
    #return obj

    
    
## functions using scipy minimize Nelder-Mead optimizer for multi layer QAOA
## Random init of beta, gamma
## returns circuit, parameters and found results
    
def QAOAoptimizeMaxCount(m,layer,Nshots,backend = BasicProvider().get_backend("basic_simulator")):

    def QAOAobjectiveFunction(x):
        size=len(x)
        size2=size//2
        return multiLayerqaoaExp(m,x[:size2],x[size2:],Nshots,backend)

    
    print("Selected device: ",backend," with ",Nshots,"shots")
    numberofparameters=2*layer
    print("Trying ",layer," layer")
    
    print("Generating inital random parameters beta and gamma")
    x0 = [random.uniform(-2*pi, 2*pi) for i in range(numberofparameters)]
    print("Starting with betas: ",x0[:(numberofparameters//2)])
    print("Starting with gammas: ",x0[(numberofparameters//2):])
    #x0 = [random.uniform(0,2*pi) for i in range(layer)]
    
    print("Optimize FIRST round with random initialisation")
    # Optimise alpha and beta using the cost function <s|H|s>
    res1 = minimize(QAOAobjectiveFunction, x0, method="Nelder-Mead")
    #print(res1)
    
    print("Optimize SECOND round with the found initialization")
    res2 = minimize(QAOAobjectiveFunction, res1.x, method="Nelder-Mead")
    #print(res2)
    bestBetas = res2.x[:(numberofparameters//2)]
    bestGammas = res2.x[(numberofparameters//2):]
    print("Best Beta", bestBetas,"Best Gamma", bestGammas)
    
    print("Now run the QAOA with the found parameters")
    vec, counts, obj, prob, qc = multiLayerqaoa(m, bestBetas, bestGammas,Nshots,backend)
    print("----------------------------------------------")
    print("Optimum = ",obj," with probability = ", prob)
    print("----------------------------------------------")
    print('Depth:', qc.depth())
    print('Gate counts:', qc.count_ops())
    
    #print(vec, obj)
    #print('Depth:', qc.depth())
    #print('Gate counts:', qc.count_ops())
    #plot_histogram(counts)
    
    return vec, counts, obj, prob, qc, res1, res2, bestBetas, bestGammas



# Set the grid size and range of parameters.

def qaoaLandscape(m,n,Nshots):
    grid_size = n
    gamma_max = math.pi
    beta_max = math.pi
    obj2 = math.inf

    # Do the grid search.
    energies = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            beta=math.pi*j/grid_size
            gamma=math.pi*i/grid_size
            vec, counts, obj, qc, prob = qaoaExp(m,beta,gamma,Nshots)
            energies[i, j] = obj   #, j * beta_max/grid_size
            if obj<obj2:
                obj2=obj
                print(beta,gamma,vec,obj2)
            
    """Plot the energy as a function of the parameters found in the grid search."""
    plt.ylabel(r"$\gamma$")
    plt.xlabel(r"$\beta$")
    plt.title("Energy as a function of parameters")
    #plt.imshow(energies, extent=(0, beta_max, gamma_max, 0))
    plt.imshow(energies, extent=(0, beta_max, 0, gamma_max))
    plt.colorbar()




## method by approximating the expectation value with the overall average of counts evaluated with the cost function

def multiLayerqaoaExpectation(m,betas,gammas,Nshots,backend = BasicProvider().get_backend("basic_simulator")):
    
    mPauli=matrixConvertInv(m)
    
    qr=QuantumRegister(len(mPauli),'q')
    cr=ClassicalRegister(len(mPauli),'c')
    qc=QuantumCircuit(qr,cr)

    
    for i in range(len(mPauli)):
        qc.h(qr[i])
    qc.barrier()
        
    for j in range(len(betas)):
        
        addGates(qr,qc,mPauli,gammas[j])
        qc.barrier()
        for k in range(len(mPauli)):
            qc.ry(betas[j],qr[k])
            #qc.barrier()
        qc.barrier()
    qc.measure(qr,cr)

    #backend=Aer.get_backend("qasm_simulator")
    #job = execute(qc, backend,shots=Nshots)
    #counts=job.result().get_counts()

    qcNew=transpile(qc, backend)
    job=backend.run(qcNew,shots=Nshots)
    counts=job.result().get_counts()

    avg = 0
    sum_count = 0

    for c in counts:
        #print(c,counts[c])
        y=counts[c]
        x=[]
        for i in c:
            x.append(int(i))

        #print(c,counts[c],x,eval_solution(x,m))
        obj=eval_solution(x,m)
        #print("Counts = ",y)
        avg += obj * y
        sum_count += y
        #expectationValue += (counts[c]*eval_solution(x,m))/Nshots
    expectationValue = avg/sum_count
    #print(expectationValue)
    #print(avg/sum_count)
    return expectationValue


# calculate expectation value through multi layer qaoa

def multiLayerqaoaExpectation1(m,betas,gammas,Nshots,backend = BasicProvider().get_backend("basic_simulator")):
    
    mPauli=matrixConvertInv(m)
    
    qr=QuantumRegister(len(mPauli),'q')
    cr=ClassicalRegister(len(mPauli),'c')
    qc=QuantumCircuit(qr,cr)

    
    for i in range(len(mPauli)):
        qc.h(qr[i])
    qc.barrier()
        
    for j in range(len(betas)):
        
        addGates(qr,qc,mPauli,gammas[j])
        qc.barrier()
        for k in range(len(mPauli)):
            qc.ry(betas[j],qr[k])
            #qc.barrier()
        qc.barrier()
    qc.measure(qr,cr)

    #backend=Aer.get_backend("qasm_simulator")
    #job = execute(qc, backend,shots=Nshots)
    #counts=job.result().get_counts()

    qcNew=transpile(qc, backend)
    job=backend.run(qcNew,shots=Nshots)
    counts=job.result().get_counts()
    
    avg = 0
    sum_count = 0

    for c in counts:
        #print(c,counts[c])
        y=counts[c]
        x=[]
        for i in c:
            x.append(int(i))

        #print(c,counts[c],x,eval_solution(x,m))
        obj=eval_solution(x,m)
        #print("Counts = ",y)
        avg += obj * y
        sum_count += y
        #expectationValue += (counts[c]*eval_solution(x,m))/Nshots
    expectationValue = avg/sum_count
    #print(expectationValue)
    #print(avg/sum_count)
    #return obj
    
    # pull result with highest count
    maxValue=-math.inf
    for c in counts:
        if counts[c]>maxValue:
            maxValue=counts[c]
            prob=maxValue/Nshots
    for c in counts:
        if counts[c]==maxValue:
            #print(c,counts[c])
            vec=c

    # calculate cost
    x=[]
    for i in maxString(counts):
        x.append(int(i))

    optimum=eval_solution(x,m)
    
    return vec, counts, expectationValue, prob, qc, optimum
    #return obj
    
    
    
## functions using scipy minimize Nelder-Mead optimizer for multi layer QAOA
## Random init of beta, gamma
## returns circuit, parameters and found results
## optional backend: can be executed on simulator (default) or real harware
## optonal optimizer: "Nelder-Mead" (default), "COBYLA", "SLSQP"
    
def QAOAoptimizeExpectation(m,layer,Nshots,backend = BasicProvider().get_backend("basic_simulator"),method="Nelder-Mead"):

    def QAOAobjectiveFunctionExpectation(x):
        size=len(x)
        size2=size//2
        return multiLayerqaoaExpectation(m,x[:size2],x[size2:],Nshots,backend)

    
    print("Selected device: ",backend," with ",Nshots,"shots")
    numberofparameters=2*layer
    print("Trying ",layer," layer")
    
    print("Generating inital random parameters beta and gamma")
    x0 = [random.uniform(-2*pi, 2*pi) for i in range(numberofparameters)]
    print("Starting with betas: ",x0[:(numberofparameters//2)])
    print("Starting with gammas: ",x0[(numberofparameters//2):])
    #x0 = [random.uniform(0,2*pi) for i in range(layer)]
    
    print("Optimize FIRST round with random initialisation")
    # Optimise alpha and beta using the cost function <s|H|s>
    res1 = minimize(QAOAobjectiveFunctionExpectation, x0, method=method)
    #print(res1)
    
    print("Optimize SECOND round with the found initialization")
    res2 = minimize(QAOAobjectiveFunctionExpectation, res1.x, method=method)
    #print(res2)
    bestBetas = res2.x[:(numberofparameters//2)]
    bestGammas = res2.x[(numberofparameters//2):]
    print("Best Beta", bestBetas,"Best Gamma", bestGammas)
    
    print("Now run the QAOA with the found parameters")
    vec, counts, expectationValue, prob, qc, optimum = multiLayerqaoaExpectation1(m, bestBetas, bestGammas,Nshots,backend)
    print("----------------------------------------------")
    print("Expectation value = ",expectationValue)
    print("----------------------------------------------")
    print("Optimum = ",optimum," with probability = ", prob)
    print("----------------------------------------------")
    print('Depth:', qc.depth())
    print('Gate counts:', qc.count_ops())
    
    #print(vec, obj)
    #print('Depth:', qc.depth())
    #print('Gate counts:', qc.count_ops())
    #plot_histogram(counts)
    
    return vec, counts, expectationValue, prob, qc, res1, res2, bestBetas, bestGammas, optimum


