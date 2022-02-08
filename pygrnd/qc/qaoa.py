'''Copyright 2022 JoS QUANTUM GmbH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

import networkx as nx
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from numpy import genfromtxt

import math
import numpy as np

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

def maxString(counts):
    bits=len(list(counts.keys())[0])
    probList=counts2probs(counts)
    stringList=allCombinations(bits)
    maxi=np.argmax(probList)
    return stringList[maxi]


def eval_solution(x,m):
    obj=np.matmul(np.matmul(x,m),np.transpose(x))
    return obj


def qaoa_qubo(m,ticks=10):

    bestValue=0
    bestBeta=0.0
    bestGamma=0.0
    besti=0
    bestSolution=[]
    resdraw=[]
    iterations=[]

    G = nx.Graph()
    G = nx.from_numpy_array(m)
    nqubits = len(G.nodes())
    print("Starting QAOA grid search for best solution with ",nqubits," qubits .... please wait .....")

    for i in range(ticks):
        print("Iteration: ",i)
        print(" --------------------------- ")
        beta=2*math.pi*i/ticks

        for j in range(ticks):
            gamma=math.pi*j/ticks
            #print("gamma = ",gamma)
            #circqaoa=qaoa_circ(G, beta, gamma, ticks)
            
            qr=QuantumRegister(nqubits,'q')
            qc=QuantumCircuit(qr)

            # initial_state
            for i in range(0, nqubits):
                qc.h(i)
            qc.barrier()

            for irep in range(0, ticks):

                # problem unitary
                for pair in list(G.edges()):
                    if (pair[0] != pair[1]):
                        qc.rzz(2 * gamma, pair[0], pair[1])
                qc.barrier()

                # mixer unitary
                for i in range(0, nqubits):
                    qc.rx(2 * beta, i)
                qc.barrier()

            qc.measure_all()

            backend = Aer.get_backend('qasm_simulator')
            job = execute(qc, backend,shots=10000)
            result = job.result()
            counts=result.get_counts()
            

            x=[]
            for k in maxString(counts):

                x.append(int(k))


            res=eval_solution(x,m)
            resdraw.append(res)
            iterations.append(j)

            if res<bestValue:
                bestValue=res
                bestSolution=x
                bestBeta=beta
                bestGamma=gamma
                besti=j
                print(i,j,res,beta,gamma)
        print(eval_solution(x,m),x, max(counts2probs(counts)))


    print("Best Value    = ",bestValue)
    print("Best Solution = ",bestSolution)
    print("Best Beta     = ",bestBeta)
    print("Best Gamma    = ",bestGamma)
    print("Best iteration= ",besti)
    print("Circuit depth = ",qc.depth())
    print('Gate counts   = ',qc.count_ops())
    qc.draw(output='mpl')

    return bestValue, bestSolution, resdraw, iterations, bestBeta, bestGamma
