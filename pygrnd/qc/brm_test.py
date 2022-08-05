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
       the classical evaluation with the result of the statevector simulation.
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
