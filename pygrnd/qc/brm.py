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

import math
import numpy as np
import qiskit
import networkx as nx

## import qiskit to build circuits
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import U3Gate
from qiskit import execute
from qiskit import Aer




def brm(nodes, edges, probsNodes, probsEdges, model2gate=False):

    ## input:
    #  Risk item list e.g.  nodes = ['0','1']
    #  Correlation risk e.g. edges=[('0','1')] # correlations
    #  probsNodes={'0':0.1,'1':0.1} # intrinsic probs
    #  probsEdges={('0','1'):0.2} # transition probs
    #  output: either circuit (model2gate=False) OR gate (model2gate=True)

    qr=QuantumRegister(len(nodes),'q')
    circuitname = QuantumCircuit(qr)
    # now find RI which cannot be triggered by transitions and put uncontrolled u3 gates in for them
    
    
    #import numpy as np
    mat=np.zeros((len(nodes),len(nodes)))
    for i in range(len(nodes)):
        mat[i][i]=probsNodes[nodes[i]]
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            if (nodes[i],nodes[j]) in probsEdges:
                mat[i][j]=probsEdges[(nodes[i],nodes[j])]
    #print(mat)
    
    
    for x in range(len(nodes)):
        cx=0
        cv=[]
    #       print("checking col",x)
        for y in range(len(nodes)):
            if mat[y,x] !=0 and x>y:
                cx=1
                cv.append(y)
        if cx==0:
            circuitname.u(2*math.asin(math.sqrt(mat[x,x])),0,0,qr[x])
        else:
            if len(cv)>1: # this RI is triggered by more than one other RI. The triggering RI are in the list "cv"
                print("NOTE: Item",x,"is triggered by more than one other RI!")
                #print(cv)
                controllist=[]
                for i in range(len(cv)):
                    controllist.append(qr[cv[i]])
                controllist.append(qr[y])
                for i in range(2**len(cv)):
                    cts = format(i, "0"+str(len(cv))+"b")
                    #print(cts)
                    if i==0:
                        p = mat[y,y]
                    else:
                        p=1
                        pbef=0
                        print("ITEM:")
                        for j in range(len(cv)):
                            if cts[j]=="1":
                                #print("mat[y,j]",mat[j,y])
                                p=p*(1-pbef)*mat[j,y]
                                pbef=mat[j,y]

                    circuitname.append(U3Gate(2*math.asin(math.sqrt(p)),0,0).control(num_ctrl_qubits=len(cv),ctrl_state=cts),controllist)
    # Here we can insert a loop that goes through all 2**len(cv) combinations of possibilities to control the RI and calculate the probability and put in a multiply controlled U3-gate 
            if len(cv)==0:
                print("there's an empty risk item ...???")
            else:
                circuitname.append(U3Gate(2*math.asin(math.sqrt(mat[x,x])),0,0).control(num_ctrl_qubits=1,ctrl_state='0'),[qr[cv[0]],qr[x]])
                ptrig = mat[cv[0],x] + (1-mat[cv[0],x])*mat[x,x]
                circuitname.append(U3Gate(2*math.asin(math.sqrt(ptrig)),0,0).control(num_ctrl_qubits=1,ctrl_state='1'),[qr[cv[0]],qr[x]])
    
    if model2gate==True:
        gate=circuitname.to_gate()
        gate.label="BRM"
        return gate, mat
    
    if model2gate==False:
        return circuitname, mat

## classical calculation of probabilities

def findAllParents(edges, currentNode):
    res=[]
    for e in edges:
        if e[1]==currentNode:
            res.append(e[0])
    return res
    
    
# Find all direct parents of a node that are not assigned yet.
def findUnassignedParents(edges, currentNode, configSoFar):
    res=[]
    for e in edges:
        if e[1]==currentNode and not(e[0] in configSoFar):
            res.append(e[0])
    return res
    
def findUnassignedNodes(nodes, configSoFar):
    res=[]
    for n in nodes:
        if not(n in configSoFar):
            res.append(n)
    return res
    
    
# Find an unassigned node that has only assigned parents.
def findCandidate(nodes, edges, configSoFar):
    unassignedNodes=findUnassignedNodes(nodes,configSoFar)
    for n in unassignedNodes:
        if len(findUnassignedParents(edges,n,configSoFar))==0:
            return n
    return []
    
    
# nodes = ['0','1','2']
# edges = [ ['0','1'],['0','2']]
# probsNodes = {'0':0.1, '1':0.2}
# probsEdges = {['0','1']=0.1, ['0','2']=0.2}
def calculateProbConfiguration(nodes, edges, probsNodes, probsEdges, probSoFar, wantedConfig, configSoFar):
    candidate=findCandidate(nodes, edges, configSoFar)
    #print("candidate=", candidate)
    if len(candidate)==0:
        return probSoFar
    parents=findAllParents(edges,candidate)

    # Calculate the probability that the candidate node is zero.
    probCandidateIsZero=1-probsNodes[candidate]
    targetCandidate=wantedConfig[candidate]
    for p in parents:
        if configSoFar[p]==1:
            probCandidateIsZero=probCandidateIsZero*(1-probsEdges[(p,candidate)])
    if targetCandidate==0:
        newProb=probSoFar*probCandidateIsZero
    else:
        newProb=probSoFar*(1-probCandidateIsZero)
    newDict={candidate:targetCandidate}
    return calculateProbConfiguration(nodes, edges, probsNodes, probsEdges, newProb, wantedConfig, {**configSoFar, **newDict})
    
 
 
def allBinaryWords(n):
    if n==1:
        return ['0','1']
    else:
        res=[]
        for b in allBinaryWords(n-1):
            res.append(b+'0')
            res.append(b+'1')
        return res
        
        
def modelProbabilities(nodes,edges,probsNodes,probsEdges):
    # calculateProbConfiguration(nodes, edges, probsNodes, probsEdges, probSoFar, wantedConfig, configSoFar)

    words=allBinaryWords(len(nodes))
    summe=0
    states=[]
    for b in words:
        buffer={}
        for i in range(len(nodes)):
            buffer[nodes[i]]=int(b[i])
        prob=calculateProbConfiguration(nodes, edges, probsNodes, probsEdges, 1, buffer, {})
        states.append(prob)
        summe=summe+prob
        #print(buffer,"-> prob=",round(prob,3))
        
    # print(summe)
    return states, summe
