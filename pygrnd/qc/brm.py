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
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import U3Gate

def brm(nodes, edges, probsNodes, probsEdges, model2gate=False):
    """input:
         Risk item list e.g.  nodes = ['0','1']
         Correlation risk e.g. edges=[('0','1')] # correlations
         probsNodes={'0':0.1,'1':0.1} # intrinsic probs
         probsEdges={('0','1'):0.2} # transition probs
         output: either circuit (model2gate=False) OR gate (model2gate=True) and the 
                 matrix with the probabilities of the nodes and the edges
    """

    qr=QuantumRegister(len(nodes),'q')
    qc=QuantumCircuit(qr)
    
    # Turn probabilities for nodes and edges into matrix form.
    mat=np.zeros((len(nodes),len(nodes)))
    for i in range(len(nodes)):
        mat[i][i]=probsNodes[nodes[i]]
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            if (nodes[i],nodes[j]) in probsEdges:
                mat[i][j]=probsEdges[(nodes[i],nodes[j])]
    
    for x in range(len(nodes)):
        cx=False
        cv=[]
        for y in range(len(nodes)):
            if mat[y,x] !=0 and x>y:
                cx=True
                cv.append(y)
        if cx==False:
            # This risk item is not triggered by transitions. Just put an uncontrolled gate in for it.
            qc.u(2*math.asin(math.sqrt(mat[x,x])),0,0,qr[x])
        else:
            if len(cv)>1: 
                # This risk item is triggered by more than one other risk item. The triggering risk item are in the list "cv"
                print("NOTE: Risk item",x,"is triggered by more than one other risk item.")
                controllist=[]
                for i in range(len(cv)):
                    controllist.append(qr[cv[i]])
                controllist.append(qr[y])
                # Here we can insert a loop that goes through all 2**len(cv) combinations of possibilities to control 
                # the risk item and calculate the probability and put in a multiply controlled U3-gate.
                for i in range(2**len(cv)):
                    cts = format(i, "0"+str(len(cv))+"b")
                    if i==0:
                        p = mat[y,y]
                    else:
                        p=1
                        pbef=0
                        for j in range(len(cv)):
                            if cts[j]=="1":
                                p=p*(1-pbef)*mat[j,y]
                                pbef=mat[j,y]
                    qc.append(U3Gate(2*math.asin(math.sqrt(p)),0,0).control(num_ctrl_qubits=len(cv),ctrl_state=cts),controllist)
            elif len(cv)==0:
                print("Note: There is an empty risk item.")
            else:
                # This risk item is triggered by one risk item.
                qc.append(U3Gate(2*math.asin(math.sqrt(mat[x,x])),0,0).control(num_ctrl_qubits=1,ctrl_state='0'),[qr[cv[0]],qr[x]])
                ptrig = mat[cv[0],x] + (1-mat[cv[0],x])*mat[x,x]
                qc.append(U3Gate(2*math.asin(math.sqrt(ptrig)),0,0).control(num_ctrl_qubits=1,ctrl_state='1'),[qr[cv[0]],qr[x]])
    
    if model2gate==True:
        gate=qc.to_gate()
        gate.label="BRM"
        return gate, mat
    else:
        return qc, mat

def findAllParents(edges, currentNode):
    res=[]
    for e in edges:
        if e[1]==currentNode:
            res.append(e[0])
    return res
    
def findUnassignedParents(edges, currentNode, configSoFar):
    """Find all direct parents of a node that are not assigned yet.
    """
    res=[]
    for e in edges:
        if e[1]==currentNode and not(e[0] in configSoFar):
            res.append(e[0])
    return res

def findUnassignedNodes(nodes, configSoFar):
    """Find all nodes that are not assigned yet.
    """
    res=[]
    for n in nodes:
        if not(n in configSoFar):
            res.append(n)
    return res
    
def findCandidate(nodes, edges, configSoFar):
    """Find an unassigned node that has only assigned parents.
    """
    unassignedNodes=findUnassignedNodes(nodes,configSoFar)
    for n in unassignedNodes:
        if len(findUnassignedParents(edges,n,configSoFar))==0:
            return n
    return []
    
def calculateProbConfiguration(nodes, edges, probsNodes, probsEdges, probSoFar, wantedConfig, configSoFar):
    """Calculate the probability of a single configuration.
    """
    candidate=findCandidate(nodes, edges, configSoFar)
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

class NodeOfEdgeNotFoundException(Exception):
    pass    

class NodeHasNoProbability(Exception):
    pass

class EdgeHasNoProbability(Exception):
    pass

def checkGraphConsistency(nodes, edges, probsNodes, probsEdges):
    """Make sure that all edges consist of nodes that are
       actually in the list of nodes. This also makes
       exception messages more readable.
    """
    for n in nodes:
        if not(n in probsNodes):
            raise NodeHasNoProbability("node "+n+" has no probability")
    for e in edges:
        if not ((e[0],e[1]) in probsEdges):
            raise EdgeHasNoProbability("edge "+str(e)+" has no probability")
        if not(e[0] in nodes):
            raise NodeOfEdgeNotFoundException("node "+e[0]+" of edge "+e+" not in list of nodes")
        if not(e[1] in nodes):
            raise NodeOfEdgeNotFoundException("node "+e[1]+" of edge "+str(e)+" not in list of nodes")

def modelProbabilities(nodes, edges, probsNodes, probsEdges, checkGraph=False):
    """Compute the list of probabilities for all possible configurations of a Business Risk Model.
       The order of the configurations is 0...0, 0...01, etc. The function also returns the sum
       of the probabilities as a consistency check.

       Example:        
       nodes = ['0', '1', '2', '3']
       edges = [ ('0','1'), ('1','2'), ('2','3')]
       probsNodes = {'0':0.1, '1':0.2, '2':0.1, '3':0.0}
       probsEdges = {('0','1'):0.1, ('1','2'):0.2, ('2','3'):0.1}
       modelProbabilities(nodes, edges, probsNodes, probsEdges, checkGraph=True)        
    """
    if checkGraph==True:
        checkGraphConsistency(nodes, edges, probsNodes, probsEdges)
    
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
        
    return states, summe

