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

import random
import pygrnd
from qiskit import execute
from qiskit import Aer
from qiskit.quantum_info import hellinger_fidelity
from pygrnd.qc.helper import *
from pygrnd.qc.brm import *
from pygrnd.qc.brm_oracle import *
from pygrnd.qc.QAE import *

def calculateDiffProbabilities(probsClassical, probsStatevector):
    '''Given are two dictionaries of probabilities. Calculate the difference
       of the probabilities.
    '''
    diffProbabilities=0
    for p in probsClassical:
        pClassical=probsClassical[p]
        pStatevector=0
        if p in probsStatevector:
            pStatevector=probsStatevector[p]
        diffProbabilities=diffProbabilities+abs(pClassical-pStatevector)
    return diffProbabilities

def randomRiskModel(numberNodes):
    '''Generate a random risk model. This is a graph without a cycle.
    '''
    probsNodes={}
    probsEdges={}
    for i in range(numberNodes):
        probsNodes[str(i)]=random.random()
        for j in range(i+1,numberNodes):
            probsEdges[(str(i),str(j))]=random.random()
    nodes=[]
    for n in probsNodes:
        nodes.append(n)
    edges=[]
    for e in probsEdges:
        edges.append(e)
    return nodes, edges, probsNodes, probsEdges

def evaluateDifferenceRandomRiskModel(numberNodes):
    '''Generate a random business risk model and compare the probabilities from
       the exact classical evaluation with the result of the statevector simulation.
    '''
    
    # Get random model.
    nodes, edges, probsNodes, probsEdges = randomRiskModel(numberNodes)

    # Classical evaluation.
    probs, sumProbs = modelProbabilities(nodes, edges, probsNodes, probsEdges)
    probsClassical={}
    for i in range(len(probs)):
        probsClassical[format(i, "0"+str(len(nodes))+"b")]=probs[i]

    # Statevector simulation.
    rm, mat = brm(nodes, edges, probsNodes, probsEdges) 
    backend = Aer.get_backend('statevector_simulator')
    job = execute(rm, backend)
    v=np.asarray(job.result().get_statevector())
    probsStatevector={}
    for i in range(len(v)):
        probsStatevector[format(i, "0"+str(len(nodes))+"b")]=abs(v[i])**2
    
    # Comparison
    return calculateDiffProbabilities(probsClassical, probsStatevector)

#
# Run the check for 200 random risk models with 4 nodes.
#
# numberNodes=4
# print("processing risk model with #nodes=",numberNodes)
# for i in range(200):
#     buffer=evaluateDifferenceRandomRiskModel(numberNodes)
#     if buffer>0.00001:
#         print("Error! Deviation is", buffer)
#

def evaluateDifferenceRandomRiskModelMonteCarlo(numberNodes, roundsMonteCarlo):
    '''Generate a random business risk model and compare the probabilities from
       the exact classical evaluation with the result of a Monte Carlo simulation
       with the given number of rounds.
    '''
    # Get a random model.
    nodes, edges, probsNodes, probsEdges=randomRiskModel(numberNodes)

    # Calculate exact solution.
    probs, sumProbs = modelProbabilities(nodes,edges,probsNodes,probsEdges)
    probsExact={}
    for i in range(len(probs)):
        probsExact[format(i, "0"+str(len(nodes))+"b")]=probs[i]

    # Monte Carlo simulation
    result=evaluateRiskModelMonteCarlo(nodes, edges, probsNodes, probsEdges, roundsMonteCarlo)

    probsMonte={}
    for i in range(2**len(nodes)):
            cts=format(i, "0"+str(len(nodes))+"b")
            if cts in result:
                probsMonte[cts]=result[cts]/roundsMonteCarlo

    return calculateDiffProbabilities(probsExact, probsMonte)

#
# Run the check for 200 random risk models with 2 nodes. Some of the Monte Carlo
# simulations might have a difference that is bigger than 0.1.
#
# numberNodes=2
# roundsMonteCarlo=1000
# for i in range(200):
#     diff=evaluateDifferenceRandomRiskModelMonteCarlo(numberNodes, roundsMonteCarlo)
#     if diff>0.1:
#         print("diff is",diff)


#
# The following tests are for the Business Risk Models with modification qubits.
#

def getModelWithSpecificModification(nodes, edges, probsNodes, probsEdges, probsNodesModified, probsEdgesModified, nodeMapping, edgeMapping, necessaryBits, modString):
        ''' Get the model parameters if we choose the modification
            that is defined by the modString. The modString should be found in nodeMapping oder edgeMapping.
        '''
        nodes2=[]
        edges2=[]
        probsNodes2={}
        probsEdges2={}
        for n in nodes:
            nodes2.append(n)
            if (n in nodeMapping) and (nodeMapping[n]==modString):
                probsNodes2[n]=probsNodesModified[n]
            else:
                probsNodes2[n]=probsNodes[n]
        for e in edges:
            edges2.append(e)
            if (e in edgeMapping) and (edgeMapping[e]==modString):
                probsEdges2[e]=probsEdgesModified[e]
            else:
                probsEdges2[e]=probsEdges[e]
        return nodes2, edges2, probsNodes2, probsEdges2

def getFidelitiesAllCombinationsTunableModel(nodes, edges, probsNodes, probsEdges, probsNodesModified, probsEdgesModified):
    """ Take a model with modifications and try out all settings of the modification qubits. Compare
        it with the normal model without modification qubits that is constructed for each case. Return
        a dictionary with modification settings and the corresponding hellinger fidelity. This fidelity
        should be high in all cases. Calculated with 100000 shots.
    """
    # Collect all the hellinger fidelities as dictionary.
    res={}
    brmMod, mat, nodeMapping, edgeMapping, necessaryBits=brmWithModifications(nodes, edges, probsNodes, probsEdges, probsNodesModified, probsEdgesModified, model2gate=True)
    modStrings=[]
    for n in nodeMapping:
        modStrings.append(nodeMapping[n])
    for e in edgeMapping:
        modStrings.append(edgeMapping[e])

    for modString in modStrings:
        # Run the modified circuit with appropriate setting of the tuning qubits.
        qt=QuantumRegister(necessaryBits,'t')
        qr=QuantumRegister(len(nodes),'q')
        cr=ClassicalRegister(len(nodes),'c')
        qc=QuantumCircuit(qt,qr,cr)
        for i in range(len(modString)):
            if modString[::-1][i]=='1':
                qc.x(qt[i])
        qc.append(brmMod,list(qt)+list(qr))
        qc.measure(qr,cr)
        backend_qasm=Aer.get_backend("qasm_simulator")
        job=execute(qc,backend=backend_qasm,shots=100000)
        countsMod=job.result().get_counts()

        # Run the normal circuit when the probs are filtered classically in advance.
        nodes2, edges2, probsNodes2, probsEdges2=getModelWithSpecificModification(nodes, edges, probsNodes, probsEdges, probsNodesModified, probsEdgesModified, nodeMapping, edgeMapping, necessaryBits, modString)
        brmClassic,mat=brm(nodes2, edges2, probsNodes2, probsEdges2, model2gate=True)
        qr=QuantumRegister(len(nodes),'q')
        cr=ClassicalRegister(len(nodes),'c')
        qc=QuantumCircuit(qr,cr)
        qc.append(brmClassic,qr)
        qc.measure(qr,cr)
        backend_qasm=Aer.get_backend("qasm_simulator")
        job=execute(qc,backend=backend_qasm,shots=100000)
        countsClassic=job.result().get_counts()
        res[modString]=hellinger_fidelity(countsMod,countsClassic)
    return res

#
# Generate a series of random risk models with randomly modified probabilities. Compare
# all possible modifications of such a model with the modified normal risk model
# without tuning qubits and report the minimum fidelity of all modifications.
#
#
# numNodes=3
# for i in range(10):
#     nodes, edges, probsNodes, probsEdges = randomRiskModel(numNodes)
#     nodes2, edges2, probsNodesModified, probsEdgesModified = randomRiskModel(numNodes)
#     brmMod, mat, nodeMapping, edgeMapping, necessaryBits=brmWithModifications(nodes, edges, probsNodes, probsEdges, probsNodesModified, probsEdgesModified, model2gate=True)
#     hellinger_fids=getFidelitiesAllCombinationsTunableModel(nodes, edges, probsNodes, probsEdges, probsNodesModified, probsEdgesModified)
#     minFid=1.0
#     for h in hellinger_fids:
#         #print(h,"->",hellinger_fids[h])
#         minFid=min(minFid,hellinger_fids[h])
#     print("minimum fidelity:",minFid)
