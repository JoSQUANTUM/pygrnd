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

#
# Standard QAE and parallel QAE (with and without intermediate resets).
# 
# See "Error Resilient Quantum Amplitude Estimation from Parallel Quantum Phase Estimation"
# https://arxiv.org/pdf/2204.01337.pdf

import math, cmath, random
import numpy as np
from math import pi
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer
from qiskit.circuit.library import QFT, HGate, XGate, UGate, SGate, ZGate, IGate
from qiskit.circuit.random import random_circuit
import itertools
from pygrnd.qc.helper import *
#
# Turn a number into binary string representation with b bits.
#
def num2bin(n, b):
    if b==0:
        return ''
    else:
        return num2bin((n-(n%2))//2,b-1)+str(n%2)

#
# Turn a binary string of 0 and 1 into an integer.
#
def bin2num(b):
    if len(b)==0:
        return 0
    else:
        return int(b[-1])+2*bin2num(b[:-1])

#
# Create a list of all binary words with given length.
#
def allBits(size):
    if size==1:
        return ['0','1']
    buffer=[]
    for s in allBits(size-1):
        buffer.append(s+'0')
        buffer.append(s+'1')
    return buffer

#
# Turn a binary result of a QAE to the corresponding probability value.
#
def bit2prob(bits):
    summe=bin2num(bits)
    res=summe/2**(len(bits)-1)
    theta_estim=math.pi*res/2.0
    prob_estim=math.sin(theta_estim)*math.sin(theta_estim)
    return prob_estim

#
# Turn counts of a histogram to relative values.
#
def counts2probs(counts):
    summe=0
    for c in counts:
        summe=summe+counts[c]
    probs={}
    for c in counts:
        prob=round(bit2prob(c),4)
        if prob in probs:
            probs[prob]=probs[prob]+counts[c]/summe
        else:
            probs[prob]=counts[c]/summe
    return probs

#
# Transform histogram with binary results of a QAE 
# to a histogram with the corresponding probabilities.
#
def getHistogramProbabilityValues(counts):
    precision=len(random.choice(list(counts.keys())))
    counts_sorted={}
    for b in allBits(precision):
        if b in counts:
            counts_sorted[b]=counts[b]
        else:
            counts_sorted[b]=0
    return counts2probs(counts_sorted)

#
# Create Grover operator as custom gate.
# Targets (e.g. ['0111','0101']) are binary strings. 
# No inputs should appear more than once.
# The number of qubits must match the length of each target string.
#
def constructGroverOperator(model, targets):
    numberQubits=model.num_qubits
    qr=QuantumRegister(numberQubits,'q')
    qc=QuantumCircuit(qr)

    # mark the good states sequentially
    for tReverse in targets:
        t=tReverse[::-1]
        lastBit=t[-1]
        if lastBit=='0':
            qc.x(qr[numberQubits-1])
        if numberQubits>1:
            qc.append(ZGate().control(num_ctrl_qubits=numberQubits-1,ctrl_state=t[:-1][::-1]),qr)
        else:
            qc.z(qr[0])
        if lastBit=='0':
            qc.x(qr[numberQubits-1])

    # Inverse operation
    qc.append(model.inverse(),qr)

    # Mark 0 with -1, include scalar -1 in front of formula
    qc.x(qr[0])
    qc.z(qr[0])
    qc.x(qr[0])
    qc.z(qr[0])
    qc.x(qr[0])
    if numberQubits>1:
        qc.append(ZGate().control(num_ctrl_qubits=numberQubits-1,ctrl_state='0'*(numberQubits-1)),qr[1:]+[qr[0]])
    else:
        qc.z(qr[0])
    qc.x(qr[0])

    # Normal operation
    qc.append(model,qr)

    grover=qc.to_gate()
    grover.label="Grover"

    return grover

#
# Collect the probability for all targets and compare these to the non-trivial eigenvalues
# of the constructed Grover operator. All printed outputs should be the same. Just for
# verifying the construction.
#
def verifyGroverOperator(model, targets):
    numberQubits=model.num_qubits

    # Sum up the probabilities from the model operator.
    qr=QuantumRegister(numberQubits,'q')
    qc=QuantumCircuit(qr)
    qc.append(model,qr)

    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    v=np.asarray(job.result().get_statevector())
    sumProbs=0
    for i in range(len(v)):
        if num2bin(i,numberQubits) in targets:
            sumProbs=sumProbs+abs(v[i])**2
    # This is the resulting probability from the statevector simulation of the model.
    print("sum of probabilities of good states=",sumProbs)

    # Create the Grover operator and check the non-trivial eigenvalues.
    qr=QuantumRegister(numberQubits,'q')
    qc=QuantumCircuit(qr)
    qc.append(constructGroverOperator(model, targets),qr)

    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    u=job.result().get_unitary()
    ev,ev2=np.linalg.eig(u)
    for x in ev:
        angle=np.angle(x)
        prob=math.sin(angle/2)**2
        if abs(prob-1)>0.01 and abs(prob-0)>0.01:
            print("probabilty from non-trivial eigenvalue of Grover operator:",prob)


#
# Return the eigenvectors that correspond to the non-trivial eigenvalues from a unitary operator.
#
def getGoodEigenvaluesUnitary(u):
    ev,ev2=np.linalg.eig(u)

    # Set the result buffer to the first eigenvector.
    goodEigenvalueIndex=0
    v=np.transpose(ev2)[goodEigenvalueIndex]
    v2=np.matmul(u,v)-ev[goodEigenvalueIndex]*v
    foundGoodOne=False

    goodIndices=[]
    for i in range(len(ev)):
        if not(abs(ev[i]-1)<0.0001 or abs(ev[i]+1)<0.0001):
            angle=np.angle(ev[i])
            prob=math.sin(angle/2)**2
            goodEigenvalueIndex=i
            print("found non-trivial eigenvalue",ev[i],"at position ",i,"and this is prob",prob)
            foundGoodOne=True
            goodIndices.append(i)

    if not(foundGoodOne):
        print("WARNING: Did not find eigenvalue except for +1 and -1")

    results=[]

    for goodEigenvalueIndex in goodIndices:
        v=np.transpose(ev2)[goodEigenvalueIndex]
        v2=np.matmul(u,v)-ev[goodEigenvalueIndex]*v

        if abs(np.matmul(np.conjugate(np.transpose(v2)),v2))**2>0.0001:
            print("WARNING: Eigenvalue equation does not hold!")

        results.append(v)
    return results

#
# Return the eigenvectors that correspond to the non-trivial eigenvalues from the Grover
# operator that is constructed from the model gate and the list of good states.
#
def getGoodEigenvaluesGroverOperator(modelGate, goodStates):
    numberQubits=modelGate.num_qubits

    qr=QuantumRegister(numberQubits,'q')
    qc=QuantumCircuit(qr)
    qc.append(constructGroverOperator(modelGate, goodStates),qr)

    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    u=job.result().get_unitary()
    return getGoodEigenvaluesUnitary(u)

#
# Add a qubit on top to get the eigenvector in the form can be compared
# directly to the approximation. Note that the approximation circuit
# has an extra qubit for creating the superposition.
#
def getEmbeddedVector(v):
    numberQubits=int(math.log(len(v))/math.log(2))
    qr=QuantumRegister(numberQubits+1,'q')
    qc=QuantumCircuit(qr)
    qc.initialize(v,qr[1:])
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    return np.asarray(job.result().get_statevector())

def getAllEmbeddedVectors(modelGate, goodStates):
    eigenvectors=getGoodEigenvaluesGroverOperator(modelGate,goodStates)

    embeddedEigenvectors=[]
    for v in eigenvectors:
        w=getEmbeddedVector(v)
        #print(w)
        embeddedEigenvectors.append(w)
    return embeddedEigenvectors

#
# Construct the gate v that generates the good state. 
# This can be only one state in the computational basis.
#
def createVgate(goodState):
    numberQubits=len(goodState)
    qr=QuantumRegister(numberQubits,'q')
    qc=QuantumCircuit(qr)
    for i in range(numberQubits):
        if goodState[numberQubits-i-1]=='1':
            qc.x(qr[i])
    gate=qc.to_gate()
    gate.label='vGate'
    return gate

#
# The P gate marks the good states with -1.
#
def createPgate(target):
    numberQubits=len(target)
    qr=QuantumRegister(numberQubits,'q')
    qc=QuantumCircuit(qr)
    t=target[::-1]
    lastBit=t[-1]
    if lastBit=='0':
        qc.x(qr[numberQubits-1])
    qc.append(ZGate().control(num_ctrl_qubits=numberQubits-1,ctrl_state=t[:-1][::-1]),qr)
    if lastBit=='0':
        qc.x(qr[numberQubits-1])
    gate=qc.to_gate()
    gate.label='pGate'
    return gate

#
# The S gate applies the phase i to all states except for the good state.
#
def createSgate(target):
    numberQubits=len(target)
    qr=QuantumRegister(numberQubits,'q')
    qc=QuantumCircuit(qr)
    t=target[::-1]
    lastBit=t[-1]
    qc.s(qr[0])
    qc.x(qr[0])
    qc.s(qr[0])
    qc.x(qr[0])
    if lastBit=='0':
        qc.x(qr[numberQubits-1])
    qc.append(SGate().inverse().control(num_ctrl_qubits=numberQubits-1,ctrl_state=t[:-1][::-1]),qr)
    if lastBit=='0':
        qc.x(qr[numberQubits-1])
    gate=qc.to_gate()
    gate.label='sGate'
    return gate

#
# The approximation of the eigenstate of the Grover operator for one good state.
# The approximation is better if the probability of the good state is small.
#
def generateStateApproximation(modelGate, goodState):
    vGate=createVgate(goodState)
    pGate=createPgate(goodState)
    sGate=createSgate(goodState)

    qr=QuantumRegister(1+len(goodState),'q')
    qc=QuantumCircuit(qr)
    qc.h(qr[0])
    qc.append(modelGate.control(num_ctrl_qubits=1,ctrl_state='0'),qr)

    qc.append(vGate.control(num_ctrl_qubits=1,ctrl_state='1'),qr)

    qc.h(qr[0])
    qc.append(pGate.control(num_ctrl_qubits=1,ctrl_state='1'),qr)
    qc.h(qr[0])

    # Choose between the approximation and its complex conjugated version.
    #qc.append(sGate.inverse().control(num_ctrl_qubits=1,ctrl_state='0'),qr)
    qc.append(sGate.control(num_ctrl_qubits=1,ctrl_state='0'),qr)

    ep=qc.to_gate()
    ep.label='ep'
    return ep

#
# Get the best of both approximations (complex conjugate) to the eigenstate or to its inverse. Note
# that we compare embedded states as the approximation circuit has an additional qubit.
# Output is the smallest deviation of the two possible eigenvalues.
#
def compareEigenvectorsWithApproximations(model, goodState):
    embeddedEigenvectors=getAllEmbeddedVectors(modelGate, [goodState])

    qr=QuantumRegister(len(goodState)+1,'q')
    qc=QuantumCircuit(qr)
    qc.append(generateStateApproximation(modelGate,goodState),qr)

    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    v_approx=np.asarray(job.result().get_statevector())

    bestApproximationSet=False
    bestApproximation=0
    for v_exact in embeddedEigenvectors:
        dif=0
        for i in range(len(v_exact)):
            buffer=v_exact[i]-v_approx[i]
            dif=dif+abs(buffer)**2
        print("this eigenvector has diff",dif)
        if bestApproximationSet:
            bestApproximation=min(bestApproximation,dif)
        else:
            bestApproximation=dif
            bestApproximationSet=True
    return bestApproximation

#
# Standard QAE
#
def circuitStandardQAE(eigenstatePreparation, groverOperator, precision):
    numberQubitsGroverOperator=groverOperator.num_qubits
    numberQubitsEP=eigenstatePreparation.num_qubits
    numberAllQubits=numberQubitsEP+precision

    if numberQubitsGroverOperator>numberQubitsEP:
        print("Error: Register of Grover operator has more qubits than register of eigenstate preparation")

    qr=QuantumRegister(numberAllQubits,"qr")
    cr=ClassicalRegister(precision,"cr")
    qc=QuantumCircuit(qr,cr)
    qc.append(eigenstatePreparation,qr[numberAllQubits-numberQubitsEP:])

    for i in range(precision):
        qc.h(qr[precision-i-1])
        qc.append(groverOperator.control().power(2**i),[qr[precision-i-1]]+qr[numberAllQubits-numberQubitsGroverOperator:])
    qc.append(QFT(precision,do_swaps=False).inverse(),qr[:precision])

    qc.measure(qr[:precision],cr)
    return qc

#
# Standard QAE. Do not measure at the end.
#
def circuitStandardQAEnoMeasurement(eigenstatePreparation, groverOperator, precision):
    numberQubitsGroverOperator=groverOperator.num_qubits
    numberQubitsEP=eigenstatePreparation.num_qubits
    numberAllQubits=numberQubitsEP+precision

    if numberQubitsGroverOperator>numberQubitsEP:
        print("Error: Register of Grover operator has more qubits than register of eigenstate preparation")

    qr=QuantumRegister(numberAllQubits,"qr")
    qc=QuantumCircuit(qr)
    qc.append(eigenstatePreparation,qr[numberAllQubits-numberQubitsEP:])

    for i in range(precision):
        qc.h(qr[precision-i-1])
        qc.append(groverOperator.control().power(2**i),[qr[precision-i-1]]+qr[numberAllQubits-numberQubitsGroverOperator:])
    qc.append(QFT(precision,do_swaps=False).inverse(),qr[:precision])

    return qc

#
# Parallel QAE, no intermediate resets.
#
def circuitStandardParallelQAE(eigenstatePreparation, groverOperator, precision):
    numberQubitsGroverOperator=groverOperator.num_qubits
    numberQubitsEP=eigenstatePreparation.num_qubits
    numberKickbackRegisters=((2**precision)-1)
    numberAllQubits=precision+numberKickbackRegisters*numberQubitsEP

    if numberQubitsGroverOperator>numberQubitsEP:
        print("Error: Register of Grover operator has more qubits than register of eigenstate preparation")

    qr=QuantumRegister(numberAllQubits,"qr")
    cr=ClassicalRegister(precision,"cr")
    qc=QuantumCircuit(qr,cr)


    startPosition=precision
    differenceSize=numberQubitsEP-numberQubitsGroverOperator
    for i in range(precision):
        qc.h(qr[precision-i-1])
        for j in range(2**i):
            qc.append(eigenstatePreparation,qr[startPosition:startPosition+numberQubitsEP])
            qc.append(groverOperator.control(),[qr[precision-i-1]]+qr[startPosition+differenceSize:startPosition+differenceSize+numberQubitsGroverOperator])
            startPosition=startPosition+numberQubitsEP
        qc.barrier()
    qc.append(QFT(precision,do_swaps=False).inverse(),qr[:precision])

    qc.measure(qr[:precision],cr)
    return qc

#
# Parallel QAE with intermediate resets.
#
def circuitStandardParallelQAEwithResets(eigenstatePreparation, groverOperator, precision):
    numberQubitsGroverOperator=groverOperator.num_qubits
    numberQubitsEP=eigenstatePreparation.num_qubits
    numberAllQubits=numberQubitsEP+precision

    if numberQubitsGroverOperator>numberQubitsEP:
        print("Error: Register of Grover operator has more qubits than register of eigenstate preparation")

    qr=QuantumRegister(numberAllQubits,"qr")
    cr=ClassicalRegister(precision,"cr")
    qc=QuantumCircuit(qr,cr)

    differenceSize=numberQubitsEP-numberQubitsGroverOperator
    for i in range(precision):
        qc.h(qr[precision-i-1])
        for j in range(2**i):
            qc.reset(qr[precision:])
            qc.append(eigenstatePreparation,qr[precision:])
            qc.append(groverOperator.control(),[qr[precision-i-1]]+qr[precision+differenceSize:])
    qc.append(QFT(precision,do_swaps=False).inverse(),qr[:precision])

    qc.measure(qr[:precision],cr)

    return qc


def findSuitableModel(qubits, minProb, maxProb, numberStates):
    """ Return a random circuit and a list of states. The states
        have together a probability between minProb and maxProb.
        The number of states can be predefined.
    """
    numberShots=10000
    foundGoodStates=False
    goodStates=0
    goodModel=0

    while not(foundGoodStates):
        qc = random_circuit(qubits, qubits+5, measure=False)

        # Remove id gates as they are useless and the controlled 
        # version does not work. This would lead to problems when 
        # creating the controlled Grover gate.
        qr=qc.qubits
        qc2=QuantumCircuit(qr)
        for gate in qc:
            if not(gate[0]==IGate()):
                qc2.append(gate[0],gate[1])
        modelGate=qc2.to_gate()
        modelGate.label='m'

        qr=QuantumRegister(qubits,'q')
        cr=ClassicalRegister(qubits,'c')
        qc=QuantumCircuit(qr,cr)
        qc.append(modelGate,qr)
        qc.measure(qr,cr)
        backend = Aer.get_backend('qasm_simulator')

        job = execute(qc, backend, shots=numberShots)
        result=job.result()
        counts=result.get_counts()
        
        # Check all subsets of size numberStates of possible combinations of results
        # if we land in the desired region. Each relevant state should have at least 1
        # hit to be relevant.
        binaryWords=allBits(qubits)
        
        for combi in itertools.combinations(binaryWords, numberStates):
            buffer=0
            allStatesPopulated=True
            for c in combi:
                if c in counts:
                    buffer=buffer+counts[c]/numberShots
                else:
                    allStatesPopulated=False

            if buffer>minProb and buffer<maxProb and allStatesPopulated==True and len(counts)>(2**qubits-2):
                foundGoodStates=True
                goodStates=list(combi)
                goodModel=modelGate
                break

    return goodModel, goodStates

def getBitStringsForClosestBin(targetProb, bits):
    """ We have a target probability and this methods returns
        the binary encodings (results of measurements after QAE)
        of the bins that correspond closest to the target probability.
    """
    allCombos=allBits(bits)
    currentBest=allCombos[0]
    currentDiff=abs(bit2prob(allCombos[0])-targetProb)
    for x in allCombos:
        if abs(bit2prob(x)-targetProb)<currentDiff:
            currentBest=x
            currentDiff=abs(bit2prob(x)-targetProb)
    res=[currentBest]
    y=complementBitstring(currentBest)
    if not(y in res):
        res.append(y)
    return res
