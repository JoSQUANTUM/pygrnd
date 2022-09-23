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
from qiskit import Aer, execute
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates import U3Gate, XGate, ZGate
from pygrnd.qc.helper import allCombinations, addValue, addPower2, subtractValue, subtractPower2,getMinusMarkerGate
from pygrnd.qc.parallelQAE import getBitStringsForClosestBin
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import QFT

def brm(nodes, edges, probsNodes, probsEdges, model2gate=False):
    """input:
         Risk item list e.g.  nodes = ['0','1']
         Correlation risk e.g. edges=[('0','1')] # correlations
         probsNodes={'0':0.1,'1':0.1} # intrinsic probs
         probsEdges={('0','1'):0.2} # transition probs
         output: either circuit (model2gate=False) or gate (model2gate=True) and the
                 matrix with the probabilities of the nodes and the edges
    """
    qr=QuantumRegister(len(nodes),'q')
    qc=QuantumCircuit(qr)
    
    # Turn probabilities for nodes and edges into matrix form.
    mat=np.zeros((len(nodes),len(nodes)))
    for i in range(len(nodes)):
        mat[i][i]=probsNodes[nodes[i]]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if (nodes[i],nodes[j]) in probsEdges:
                mat[i][j]=probsEdges[(nodes[i],nodes[j])]
    
    # Main processing loop.
    indicesProcessed=[]
    while len(indicesProcessed)<len(nodes):
        # Find the first unprocessed node that has no unprocessed parents.
        target=None
        for i in range(len(nodes)):
            allParentsAlreadyProcessed=True
            for j in range(len(nodes)):
                if not(i==j) and (mat[j][i]!=0) and not(j in indicesProcessed):
                    allParentsAlreadyProcessed=False
            if not(i in indicesProcessed) and allParentsAlreadyProcessed==True:
                target=i
                break
        indicesProcessed.append(target)
        
        # We might have a cycle.
        if target==None:
            print("Internal error. Please check the nodes and edges.")
            return qc, mat
        
        foundControllers=False
        collectedControllerIndices=[]
        for y in range(len(nodes)):
            if mat[y,target] !=0 and not(y==target):
                foundControllers=True
                collectedControllerIndices.append(y)
        
        if foundControllers==False:
            # This risk item is not triggered by transitions. Just put an uncontrolled gate in for it.
            qc.u(2*math.asin(math.sqrt(mat[target,target])),0,0,qr[target])
        else:
            # This risk item is triggered by more than one other risk item. The triggering risk item are in the list "collectedControllerIndices"
                controllist=[]
                for i in range(len(collectedControllerIndices)):
                    controllist.append(qr[collectedControllerIndices[i]])
                controllist.append(qr[target])
                
                #
                # Iterate over all binary configurations of the control qubits and calculate the probability that
                # the target node is not triggered.
                #
                for i in range(2**len(collectedControllerIndices)):
                    cts = format(i, "0"+str(len(collectedControllerIndices))+"b")
                    
                    pTargetOff=1-mat[target,target]
                    for j in range(len(collectedControllerIndices)):
                        if cts[j]=="1":
                            pTargetOff=pTargetOff*(1-mat[collectedControllerIndices[j],target])
                            
                    # For this configuration of control qubits, turn the qubit on with
                    # the probability 1-pTargetOff.
                    qc.append(U3Gate(2*math.asin(math.sqrt(1-pTargetOff)),0,0).control(num_ctrl_qubits=len(collectedControllerIndices),ctrl_state=cts[::-1]),controllist)


    if model2gate==True:
        gate=qc.to_gate()
        gate.label="BRM"
        return gate, mat
    else:
        return qc, mat

def variationsDictionary(nodes, edges, probsNodes, probsEdges, probsNodesModified, probsEdgesModified):
    """ We can modify the probabilities of the nodes and the edges. How
        many different values are there in total for nodes and edges? Return
        a mapping for the nodes and the edges that can be modified as binary numbers.
        Keep the combination 0..0 for no modification.
    """
    numberDiffs=0
    for n in nodes:
        if not(probsNodes[n]==probsNodesModified[n]):
            numberDiffs=numberDiffs+1
    for e in edges:
        if not(probsEdges[e]==probsEdgesModified[e]):
            numberDiffs=numberDiffs+1

    # Keep 0..0 for 'no modification'
    if numberDiffs>0:
        numberDiffs=numberDiffs+1

    necessaryBits=0
    if numberDiffs>0:
        necessaryBits=math.ceil(math.log(numberDiffs)/math.log(2))
    allCombos=allCombinations(necessaryBits)

    nodeMapping={}
    edgeMapping={}
    currentPosition=1
    for n in nodes:
        if not(probsNodes[n]==probsNodesModified[n]):
            nodeMapping[n]=allCombos[currentPosition]
            currentPosition=currentPosition+1

    for e in edges:
        if not(probsEdges[e]==probsEdgesModified[e]):
            edgeMapping[e]=allCombos[currentPosition]
            currentPosition=currentPosition+1
    return nodeMapping, edgeMapping, necessaryBits

def appendDependentNode(qt, qr, qc, mat, mat2, target, collectedControllerIndices, modifiableEdges, nodes, nodeMapping, edgeMapping, necessaryBits):
    """ Iterate over all binary configurations of the control qubits
        and calculate the probability that the target node is not triggered.
    """
    controllist=[]
    for i in range(len(collectedControllerIndices)):
        controllist.append(qr[collectedControllerIndices[i]])
    controllist.append(qr[target])

    for i in range(2**len(collectedControllerIndices)):
        cts = format(i, "0"+str(len(collectedControllerIndices))+"b")

        pTargetOff=1-mat[target,target]
        for j in range(len(collectedControllerIndices)):
            if cts[j]=="1":
                pTargetOff=pTargetOff*(1-mat[collectedControllerIndices[j],target])

        # For this configuration of control qubits, turn the qubit on with
        # the probability 1-pTargetOff.
        qc.append(U3Gate(2*math.asin(math.sqrt(1-pTargetOff)),0,0).control(num_ctrl_qubits=len(collectedControllerIndices),ctrl_state=cts[::-1]),controllist)

        if nodes[target] in nodeMapping:
            pTargetOff3=1-mat2[target,target]
            for j in range(len(collectedControllerIndices)):
                if cts[j]=="1":
                    pTargetOff3=pTargetOff3*(1-mat[collectedControllerIndices[j],target])
            newValue=2*math.asin(math.sqrt(1-pTargetOff3))-2*math.asin(math.sqrt(1-pTargetOff))
            nodeString=nodeMapping[(nodes[target])]
            qc.append(U3Gate(newValue,0,0).control(num_ctrl_qubits=len(collectedControllerIndices)+necessaryBits,ctrl_state=cts[::-1]+nodeString),list(qt)+controllist)

        # We need to modify the value if we have a modified edge. Just this edge is different.
        # And only if the modified edge is active in this configuration.
        for m in modifiableEdges:
            if (cts[collectedControllerIndices.index(m[0])]=='1'):
                pTargetOff2=1-mat[target,target]
                for j in range(len(collectedControllerIndices)):
                    if cts[j]=="1":
                        if not(j==collectedControllerIndices.index(m[0])):
                            pTargetOff2=pTargetOff2*(1-mat[collectedControllerIndices[j],target])
                        else:
                            pTargetOff2=pTargetOff2*(1-mat2[collectedControllerIndices[j],target])
                    newValue=2*math.asin(math.sqrt(1-pTargetOff2))-2*math.asin(math.sqrt(1-pTargetOff))
                edgeString=edgeMapping[(nodes[m[0]],nodes[m[1]])]
                qc.append(U3Gate(newValue,0,0).control(num_ctrl_qubits=len(collectedControllerIndices)+necessaryBits,ctrl_state=cts[::-1]+edgeString),list(qt)+controllist)

def brmWithModifications(nodes, edges, probsNodes, probsEdges, probsNodesModified, probsEdgesModified, model2gate=False):
    """input:
         Risk item list e.g.  nodes = ['0','1']
         Correlation risk e.g. edges=[('0','1')] # correlations
         probsNodes={'0':0.1,'1':0.1} # intrinsic probs
         probsEdges={('0','1'):0.2} # transition probs
         probsNodesModified={'0':0.1,'1':0.1} # intrinsic probs
         probsEdgesModified={('0','1'):0.3} # transition probs
         output: Either circuit (model2gate=False) or gate (model2gate=True) and the
                 matrix with the probabilities of the nodes and the edges. Also, return
                 dictionaries along with the number of tuning qubits that determine which
                 modification of nodes and edges should be turned on.
    """

    # Consider the variations and the necessary qubits to encode them.
    nodeMapping, edgeMapping, necessaryBits = variationsDictionary(nodes, edges, probsNodes, probsEdges, probsNodesModified, probsEdgesModified)

    qt=QuantumRegister(necessaryBits,'t') # tuning parameters
    qr=QuantumRegister(len(nodes),'q')
    qc=QuantumCircuit(qt,qr)

    # Turn probabilities for nodes and edges into matrix form. Unmodified version.
    mat=np.zeros((len(nodes),len(nodes)))
    for i in range(len(nodes)):
        mat[i][i]=probsNodes[nodes[i]]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if (nodes[i],nodes[j]) in probsEdges:
                mat[i][j]=probsEdges[(nodes[i],nodes[j])]

    # Turn probabilities for nodes and edges into matrix form. Modified version.
    mat2=np.zeros((len(nodes),len(nodes)))
    for i in range(len(nodes)):
        mat2[i][i]=probsNodesModified[nodes[i]]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if (nodes[i],nodes[j]) in probsEdgesModified:
                mat2[i][j]=probsEdgesModified[(nodes[i],nodes[j])]

    # Main processing loop.
    indicesProcessed=[]
    while len(indicesProcessed)<len(nodes):
        # Find the first unprocessed node that has no unprocessed parents.
        target=None
        for i in range(len(nodes)):
            allParentsAlreadyProcessed=True
            for j in range(len(nodes)):
                if not(i==j) and (mat[j][i]!=0) and not(j in indicesProcessed):
                    allParentsAlreadyProcessed=False
            if not(i in indicesProcessed) and allParentsAlreadyProcessed==True:
                target=i
                break
        indicesProcessed.append(target)

        # We might have a cycle.
        if target==None:
            print("Internal error. Please check the nodes and edges.")
            return qc, mat

        foundControllers=False
        collectedControllerIndices=[]
        for y in range(len(nodes)):
            if mat[y,target] !=0 and not(y==target):
                foundControllers=True
                collectedControllerIndices.append(y)

        if foundControllers==False:
            # This risk item is not triggered by transitions. Just put an uncontrolled gate in for it. It can only be controlled by the
            # modification register.
            qc.u(2*math.asin(math.sqrt(mat[target,target])),0,0,qr[target])
            if nodes[target] in nodeMapping:
                # The probability must be modified if the modification setting is active.
                newValue=2*math.asin(math.sqrt(mat2[target,target]))-2*math.asin(math.sqrt(mat[target,target]))
                qc.append(U3Gate(newValue,0,0).control(num_ctrl_qubits=necessaryBits,ctrl_state=nodeMapping[nodes[target]]),qt[:]+[qr[target]])
        else:
            # This risk item is triggered by more than one other risk item. The triggering risk item are in the list "collectedControllerIndices"
            modifiableEdges=[]
            for source in collectedControllerIndices:
                if (nodes[source],nodes[target]) in edgeMapping:
                    modifiableEdges.append((source,target))
            appendDependentNode(qt, qr, qc, mat, mat2, target, collectedControllerIndices, modifiableEdges, nodes, nodeMapping, edgeMapping, necessaryBits)

    if model2gate==True:
        gate=qc.to_gate()
        gate.label="BRM"
        return gate, mat, nodeMapping, edgeMapping, necessaryBits
    else:
        return qc, mat, nodeMapping, edgeMapping, necessaryBits

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
        # Reverse bit order to be consistent with Qiskit qubit order.
        bInv=b[::-1]
        buffer={}
        for i in range(len(nodes)):
            buffer[nodes[i]]=int(bInv[i])
        prob=calculateProbConfiguration(nodes, edges, probsNodes, probsEdges, 1, buffer, {})
        states.append(prob)
        summe=summe+prob
        
    return states, summe

def processNodesMonteCarlo(nodes, edges, probsNodes, probsEdges, configSoFar):
    ''' Find all nodes without unprocessed predecessors. Then process
        these nodes and proceed recursively. This is used for one run
        of the risk model with Monte Carlo.
    '''

    # All nodes are already processed.
    if len(configSoFar)==len(nodes):
        return configSoFar

    # Find unprocessed nodes without unprocessed predecessors.
    candidates=[]
    for n in nodes:
        if not (n in configSoFar):
            onlyProcessedPrecedessors=True
            for e in edges:
                if (e[1]==n) and not(e[0] in configSoFar):
                    onlyProcessedPrecedessors=False
            if onlyProcessedPrecedessors:
                candidates.append(n)

    # Process all candidates.
    for c in candidates:
        active=0
        if random.random()<probsNodes[c]:
            active=1
        for e in edges:
            if e[1]==c:
                if configSoFar[e[0]]==1:
                    if random.random()<probsEdges[e]:
                        active=1
        configSoFar[c]=active

    return processNodesMonteCarlo(nodes, edges, probsNodes, probsEdges, configSoFar)

def getRandomConfigurationMonteCarlo(nodes, edges, probsNodes, probsEdges):
    ''' Calculate a valid configuration of the nodes of the risk model according
        to the given probabilities. Return the binary string of the configuration.
        Note that the order is reversed to be consistent with Qiskits qubit order.
    '''
    config=processNodesMonteCarlo(nodes, edges, probsNodes, probsEdges,{})
    string=''
    for n in nodes[::-1]:
        string=string+str(config[n])
    return string

def evaluateRiskModelMonteCarlo(nodes, edges, probsNodes, probsEdges, rounds):
    ''' Evaluate the risk model with the Monte Carlo method many times and
        return a dictionary of the counts of all calculated configurations.
        Note that the order is reversed to be consistent with Qiskits qubit order.
    '''
    result={}
    for i in range(rounds):
        string=getRandomConfigurationMonteCarlo(nodes, edges, probsNodes, probsEdges)
        if string in result:
            result[string]=result[string]+1
        else:
            result[string]=1
    return result

def addCostsToRiskModelCircuit(riskModelCircuit, nodes, costsNodes, sizeCostRegister):
    """ Add a cost register to a risk model circuit. The circuit might have tuning qubits.
        Also add the gates to sum up the costs of risk items that are active at the end.
    """
    # Sort out classical registers. Assume that the last len(nodes) qubits are
    # the qubits representing the risk items.
    quantumRegisters=[]
    for x in riskModelCircuit.qubits:
        if type(x)==QuantumRegister:
            quantumRegisters.append(x)
    nodeRegister=riskModelCircuit.qubits[-len(nodes):]
    # Add a register for the costs and add the controlled adders.
    costRegister=QuantumRegister(sizeCostRegister,'costs')
    riskModelCircuit.add_register(costRegister)
    for i in range(len(nodes)):
        qrTemp=QuantumRegister(sizeCostRegister)
        qcTemp=QuantumCircuit(qrTemp)
        addValue(qrTemp,qcTemp,costsNodes[nodes[i]])
        gateTemp=qcTemp.to_gate()
        riskModelCircuit.append(gateTemp.control(),[nodeRegister[i]]+list(costRegister))

def addCostsAndLimitToRiskModelCircuit(riskModelCircuit, nodes, costsNodes, limit):
    """ Add the summation of the costs to a risk model. Subtract the
        limit and set a indicator qubit when a configuration of the
        risk model results in costs that are equal or above the limit.
    """
    maxCosts=sum([costsNodes[x] for x in costsNodes])
    sizeCostRegister=math.ceil(math.log(maxCosts)/math.log(2))+1
    addCostsToRiskModelCircuit(riskModelCircuit, nodes, costsNodes, sizeCostRegister)
    costRegister=riskModelCircuit.qubits[-sizeCostRegister:]
    subtractValue(costRegister,riskModelCircuit,limit)
    limitQubit=QuantumRegister(1,"limit")
    riskModelCircuit.add_register(limitQubit)
    riskModelCircuit.append(XGate().control(num_ctrl_qubits=1,ctrl_state='0'),[costRegister[-1],limitQubit[0]])

def constructGroverOperatorForRiskModelWithLimit(riskModel, necessaryBits):
    """ The input is a risk model with cost register and indicator
        qubit at the end that shows that the limit is reached. The value of necessaryBits
        shows the size of the register for the tuning qubits of the model. This
        method constructs a Grover operator for the states that are
        equal or above the limit.
    """
    riskModelGate=riskModel.to_gate()

    numberQubits=riskModel.num_qubits
    qr=QuantumRegister(numberQubits,'q')
    qc=QuantumCircuit(qr)

    # mark the state as good if the last qubit is active
    qc.z(qr[-1])

    # Inverse operation
    qc.append(riskModelGate.inverse(),qr)

    # Mark 0 with -1, include scalar -1 in front of formula
    qc.x(qr[necessaryBits])
    qc.z(qr[necessaryBits])
    qc.x(qr[necessaryBits])
    qc.z(qr[necessaryBits])
    qc.x(qr[necessaryBits])
    qc.append(ZGate().control(num_ctrl_qubits=numberQubits-necessaryBits-1,ctrl_state='0'*(numberQubits-necessaryBits-1)),qr[necessaryBits+1:]+[qr[necessaryBits]])
    qc.x(qr[necessaryBits])

    # Normal operation
    qc.append(riskModelGate,qr)

    return qc

def getGroverOracleFromQAEoracle(numQubitsOracle, qae, resolution, targetProb):
    """ Create the a Grover type oracle from the QAE with a chosen target probability.
        The bins that correspond to the probability closest to the target are chosen
        and used to introduce a phase of -1. The first qubit is added for the
        generation of the phase.
    """

    bins=getBitStringsForClosestBin(targetProb, resolution)

    qr=QuantumRegister(numQubitsOracle+1,'q')
    qc=QuantumCircuit(qr)
    qc.x(qr[0])
    qc.append(qae,qr[1:])

    for c in bins:
        qc.append(ZGate().control(num_ctrl_qubits=resolution,ctrl_state=c),qr[1:resolution+1]+[qr[0]])

    qc.append(qae.inverse(),qr[1:])
    qc.x(qr[0])
    return qc

def circuitGroverOverQAE(iterations, numQubits, qae, resolution, requiredQubits, targetProb):
    """ Construct a Grover search for a QAE constructed from a risk model with costs and a limit.
        The target probability is replaced by the closest possible value of the QAE with the
        resolution.
    """
    sandwich=getGroverOracleFromQAEoracle(numQubits, qae, resolution, targetProb)

    minus=getMinusMarkerGate(requiredQubits)

    qr=QuantumRegister(numQubits+1,'q')
    cr=ClassicalRegister(requiredQubits,'c')
    qc=QuantumCircuit(qr,cr)
    for i in range(requiredQubits):
        qc.h(qr[1+resolution+i])

    for j in range(iterations):
        qc.append(sandwich,qr)
        for i in range(requiredQubits):
            qc.h(qr[1+resolution+i])
        qc.append(minus,qr[1+resolution:1+resolution+requiredQubits])
        for i in range(requiredQubits):
            qc.h(qr[1+resolution+i])

    qc.measure(qr[1+resolution:1+resolution+requiredQubits],cr)
    return qc
