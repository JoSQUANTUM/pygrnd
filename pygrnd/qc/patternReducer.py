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

import json
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit.circuit.library import RGate,U3Gate,GMS,QFT,XGate,RXXGate,ZGate,PhaseGate,SwapGate,CXGate,RXGate,RYGate,RZGate, SXGate, SXdgGate, CPhaseGate, HGate,CCXGate
from qiskit import Aer
import qiskit
import math
from scipy.linalg import pinv
import numpy as np
import itertools
from pygrnd.qc.patternGenerator import insertPatternGates

def invertParameters(params):
    if type(params)==type([]):
        res=[]
        for p in params:
            res.append(-p)
        return res
    elif type(params)==type(1.1):
        return -params
    elif type(params)==type(1):
        return -params
    else:
        print("Error: Unsupported type in invertParameters:",params,type(params))

def invertPattern(pattern):
    res=[]
    for p in pattern[::-1]:
        if p[0]=='X':
            res.append(p)
        elif p[0]=='H':
            res.append(p)
        elif p[0]=='Z':
            res.append(p)
        elif p[0]=='SWAP':
            res.append(p)
        elif p[0]=='CNOT':
            res.append(p)
        elif p[0]=='CCX':
            res.append(p)
        elif p[0]=='CZ':
            res.append(p)
        elif p[0]=='S':
            res.append(["Sdg",{},p[2]])
        elif p[0]=='Sdg':
            res.append(["S",{},p[2]])
        elif p[0]=='T':
            res.append(["Tdg",{},p[2]])
        elif p[0]=='Tdg':
            res.append(["T",{},p[2]])
        elif p[0]=='SX':
            res.append(["SXdg",{},p[2]])
        elif p[0]=='SXdg':
            res.append(["SX",{},p[2]])
        elif p[0]=='P':
            res.append(["P",{'lambda':invertParameters(p[1]['lambda'])},p[2]])
        elif p[0]=='RXX':
            res.append(["RXX",{'theta':invertParameters(p[1]['theta'])},p[2]])
        elif p[0]=='GPI':
            res.append(["GPI",{'phi':invertParameters(p[1]['phi'])},p[2]])
        elif p[0]=='GPI2':
            res.append(["GPI2",{'phi':p[1]['phi']+math.pi},p[2]])
        elif p[0]=='GZ':
            res.append(["GZ",{'theta':invertParameters(p[1]['theta'])},p[2]])
        elif p[0]=='CP':
            res.append(["CP",{'lambda':invertParameters(p[1]['lambda'])},p[2]])
        elif p[0]=='RX':
            res.append(["RX",{'theta':invertParameters(p[1]['theta'])},p[2]])
        elif p[0]=='RY':
            res.append(["RY",{'theta':invertParameters(p[1]['theta'])},p[2]])
        elif p[0]=='RZ':
            res.append(["RZ",{'lambda':invertParameters(p[1]['lambda'])},p[2]])
        else:
            print("ERROR: unsupported gate in invertPattern:",p)
    return res

def patternsEquality(patternA,patternB):
    #print("patternsEquality:",patternA,patternB)
    if not(len(patternA)==len(patternB)):
        return False
    for i in range(len(patternA)):
        bufferA=patternA[i]
        bufferB=patternB[i]

        # Type must be equal.
        if not(bufferA[0]==bufferB[0]):
            return False

        # Parameters must be the same up to a certain error
        for param in bufferA[1]:
            if not(param in bufferB[1]):
                return False
            if type(bufferA[1][param])==type([]) or type(bufferB[1][param])==type([]):
                return False
            elif abs(bufferA[1][param]-bufferB[1][param])>0.000001:
                return False

        # Wrong qubits.
        if not(bufferA[2]==bufferB[2]):
            return False
    return True

def unifyPatternPhase(pattern, candidate):
    #print("unifier",pattern,candidate)
    # The reducer candidate should be longer than the concrete pattern.
    if len(pattern)>len(candidate):
        return candidate

    # The type of elements must be the same. Else it is not possible to have a match.
    for i in range(len(pattern)):
        #print(pattern[i],candidate[i])
        if not(pattern[i][0]==candidate[i][0]):
            return candidate

    # Only create a matrix entry for the candidate if a phase is of the form [1,0]
    phasesPattern=[] # like 0.1
    phasesCandidate=[] # like [1,-1]
    for i in range(len(pattern)):
        currentPattern=pattern[i]
        currentCandidate=candidate[i]
        for para in currentCandidate[1]:
            if type(currentCandidate[1][para])==type([]):
                phasesPattern.append(currentPattern[1][para])
                phasesCandidate.append(currentCandidate[1][para])

    matrix=np.array(phasesCandidate[:len(phasesPattern)])
    vector=np.array(phasesPattern)
    minv=np.array([[0]])
    invertedMatrix=np.array([[0]])
    invertedVector=np.array([[0]])
    if len(matrix)>0:
        minv=pinv(matrix)
        invertedMatrix=np.matmul(minv,matrix)
        invertedVector=np.matmul(minv,vector)
    #print("inv.matrix=",invertedMatrix)
    #print("inv.vector=",invertedVector)

    # Check that the values in pattern are reconstructed correctly.
    for i in range(len(phasesPattern)):
        currentCandidate=phasesCandidate[i]
        vectorCandidate=np.array(currentCandidate)
        #print(currentCandidate,"-=>",np.matmul(vectorCandidate,invertedVector[:len(invertedMatrix)])," and should be ",phasesPattern[i])
        if abs(np.matmul(vectorCandidate,invertedVector[:len(invertedMatrix)])-phasesPattern[i])>0.00001:
            return candidate

    # Now create the candidate with replaced values.
    modifiedCandidate=[]
    for c in candidate:
        newPhases={}
        for param in c[1]:
            if type(c[1][param])==type([]):
                if not(len(invertedMatrix)==len(c[1][param])):
                    return candidate
                replacedValue=np.matmul(np.array(c[1][param]),invertedVector[:len(invertedMatrix)])
            else:
                replacedValue=c[1][param]
            #print("param",param," is vector",c[1][param]," and should be",replacedValue)
            newPhases[param]=float(replacedValue)
        modifiedCandidate.append([c[0],newPhases,c[2]])
    return modifiedCandidate

def reducePattern(pattern, patternDataBase):

    replacers=[pattern] # The replacement with the same is always possible.

    for candidate in patternDataBase:
        candidate2=unifyPatternPhase(pattern,candidate)
        if len(candidate2)>=len(pattern):
            if patternsEquality(pattern,candidate2[:len(pattern)]): # The unified front part must be the same.
                remaining=candidate2[len(pattern):]
                replacers.append(invertPattern(remaining))

    replacersUnique=[]
    for r in replacers:
        if not(r in replacersUnique):
            replacersUnique.append(r)

    return replacersUnique

def sortedQubits(qc,qubits):
    qr=qc._qubits
    #print(qr)
    indexSet=[]
    for q in qubits:
        indexSet.append(qr.index(q))
    indexSet.sort()
    newQubits=[]
    for i in indexSet:
        newQubits.append(qr[i])
    return newQubits

def transformBlock(qc, candidate):
    pattern=[]

    # Pick out the relevant gates.
    relevantGates=[]
    for c in candidate:
        relevantGates.append(qc[c])

    # Find the relevant qubits.
    relevantQubits=[]
    for g in relevantGates:
        relevantQubits=relevantQubits+g[1]
    relevantQubits=list(set(relevantQubits))
    #print(relevantQubits)
    relevantQubits=sortedQubits(qc,relevantQubits)
    #print("relevant qubits:", relevantQubits)

    # Generate the pattern for the candidate block.
    pattern=[]
    for g in relevantGates:
        # Transform qubits to abstract version.
        touchedQubits=[]
        for q in g[1]:
            touchedQubits.append(relevantQubits.index(q))
        #print(g)
        # Transform gates to abstract version.
        if type(g[0])==qiskit.circuit.library.standard_gates.h.HGate:
            pattern.append(["H",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.standard_gates.x.XGate:
            pattern.append(["X",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.SGate:
            pattern.append(["S",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.SdgGate:
            pattern.append(["Sdg",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.TGate:
            pattern.append(["T",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.TdgGate:
            pattern.append(["Tdg",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.standard_gates.sx.SXGate:
            pattern.append(["SX",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.SXdgGate:
            pattern.append(["SXdg",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.standard_gates.z.ZGate:
            pattern.append(["Z",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.CCXGate:
            pattern.append(["CCX",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.standard_gates.x.CXGate:
            pattern.append(["CNOT",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.standard_gates.z.CZGate:
            pattern.append(["CZ",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.standard_gates.swap.SwapGate:
            pattern.append(["SWAP",{},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.standard_gates.p.CPhaseGate:
            currentParams=g[0].base_gate._params
            currentPhase=currentParams[0]
            if currentPhase>math.pi:
                currentPhase=currentPhase-2*math.pi
            pattern.append(["CP",{'lambda':currentPhase},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.standard_gates.rxx.RXXGate:
            currentParams=g[0]._params
            currentPhase=currentParams[0]
            if currentPhase>math.pi:
                currentPhase=currentPhase-2*math.pi
            pattern.append(["RXX",{'theta':currentPhase},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.U3Gate:
            currentParams=g[0]._params
            currentTheta=currentParams[0]
            currentPhi=currentParams[1]
            currentLambda=currentParams[2]
            if currentTheta>math.pi:
                currentTheta=currentTheta-2*math.pi
            if currentPhi>math.pi:
                currentPhi=currentPhi-2*math.pi
            if currentLambda>math.pi:
                currentLambda=currentLambda-2*math.pi
            pattern.append(["U3",{'theta':currentTheta,'phi':currentPhi,'lambda':currentLambda},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.RGate:
            currentParams=g[0]._params
            currentTheta=currentParams[0]
            currentPhi=currentParams[1]
            if currentTheta>math.pi:
                currentTheta=currentTheta-2*math.pi
            if currentPhi>math.pi:
                currentPhi=currentPhi-2*math.pi
            pattern.append(["R",{'theta':currentTheta,'phi':currentPhi},touchedQubits])
        elif type(g[0])==qiskit.circuit.library.standard_gates.p.PhaseGate:
            currentParams=g[0]._params
            currentPhase=currentParams[0]
            if currentPhase>math.pi:
                currentPhase=currentPhase-2*math.pi
            pattern.append(["P",{'lambda':currentPhase},touchedQubits])
            #pattern.append(["P",g[0]._params,touchedQubits])
        elif type(g[0])==qiskit.circuit.library.RXGate:
            currentParams=g[0]._params
            currentPhase=currentParams[0]
            if currentPhase>math.pi:
                currentPhase=currentPhase-2*math.pi
            pattern.append(["RX",{'theta':currentPhase},touchedQubits])
            #pattern.append(["RX",g[0]._params,touchedQubits])
        elif type(g[0])==qiskit.circuit.library.RYGate:
            currentParams=g[0]._params
            currentPhase=currentParams[0]
            if currentPhase>math.pi:
                currentPhase=currentPhase-2*math.pi
            pattern.append(["RY",{'theta':currentPhase},touchedQubits])
            #pattern.append(["RY",g[0]._params,touchedQubits])
        elif type(g[0])==qiskit.circuit.library.RZGate:
            currentParams=g[0]._params
            currentPhase=currentParams[0]
            if currentPhase>math.pi:
                currentPhase=currentPhase-2*math.pi
            pattern.append(["RZ",{'lambda':currentPhase},touchedQubits])
            #pattern.append(["RZ",g[0]._params,touchedQubits])
        else:
            print("The gate ",g, "is not supported. End procedure.")
            return []
    return pattern

def collectGates(qc, gateIndex, relevantQubitsSetOrig, activeQubitsSetOrig, blockedQubitsSetOrig, collectedGatesSoFarOrig):
    #print("start collectedGates ",gateIndex,"act=",activeQubitsSetOrig,"blk=",blockedQubitsSetOrig,"coll=",collectedGatesSoFarOrig)

    relevantQubitsSet=relevantQubitsSetOrig.copy()
    activeQubitsSet=activeQubitsSetOrig.copy()
    blockedQubitsSet=blockedQubitsSetOrig.copy()
    collectedGatesSoFar=collectedGatesSoFarOrig.copy()

    if gateIndex==len(qc):
        # end reached
        #print("end of recursion with collected",collectedGatesSoFar)
        return [collectedGatesSoFar]

    currentGate=qc[gateIndex]
    currentQubitsSet=set(currentGate[1])

    if len(set.intersection(relevantQubitsSet,currentQubitsSet))==0:
        # This gate has no relevant qubit. Ignore it.
        #print("gate",gateIndex,"is not relevant")
        return collectGates(qc,gateIndex+1,relevantQubitsSet,activeQubitsSet,blockedQubitsSet,collectedGatesSoFar)

    if len(set.intersection(activeQubitsSet,currentQubitsSet))>0 and len(set.difference(currentQubitsSet,relevantQubitsSet))>0:
        # This gate uses an already active qubit and it has an dependency to outside. Do not use it, but block the
        # active qubits.
        #print("gate",gateIndex, "uses active components and blocks some qubits")
        blockedQubitsSetNew=set.union(blockedQubitsSet,set.intersection(currentQubitsSet,relevantQubitsSet))
        return collectGates(qc,gateIndex+1,relevantQubitsSet,activeQubitsSet,blockedQubitsSetNew,collectedGatesSoFar)

    if len(set.intersection(activeQubitsSet,currentQubitsSet))==0 and len(set.difference(currentQubitsSet,relevantQubitsSet))>0:
        # This gate is not using an active qubit and has a dependency to the outside. Do not use it and do not
        # block any active qubit.
        #print("gate",gateIndex, "does not use active components and does not block but has external dependencies")
        return collectGates(qc,gateIndex+1,relevantQubitsSet,activeQubitsSet,blockedQubitsSet,collectedGatesSoFar)

    if len(set.intersection(relevantQubitsSet,currentQubitsSet))==len(currentQubitsSet) and len(set.intersection(blockedQubitsSet,currentQubitsSet))>0:
        # This gate is good, but blocked. Do not use it. Block all qubits that are used by it.
        #print("gate",gateIndex, "is fully inside relevant gates but it is blocked")
        blockedQubitsSetNew=set.union(blockedQubitsSet,set.intersection(currentQubitsSet,relevantQubitsSet))
        return collectGates(qc,gateIndex+1,relevantQubitsSet,activeQubitsSet,blockedQubitsSetNew,collectedGatesSoFar)

    if len(set.intersection(relevantQubitsSet,currentQubitsSet))==len(currentQubitsSet) and len(set.intersection(activeQubitsSet,currentQubitsSet))==0:
        # This gate is fully inside the relevant set, i.e. no external dependency. And it does not
        # use already active qubits. It might be taken or not.
        #print("gate",gateIndex,"can be considered (no active qubit touched)")
        #print("cur=",currentQubitsSet)
        #print("blk=",blockedQubitsSet)
        #print("len/len=",len(set.intersection(relevantQubitsSet,currentQubitsSet)),len(set.intersection(blockedQubitsSet,currentQubitsSet)))
        # Do not take it.
        res1=collectGates(qc,gateIndex+1,relevantQubitsSet,activeQubitsSet,blockedQubitsSet,collectedGatesSoFar)
        # Take it.
        res2=collectGates(qc,gateIndex+1,relevantQubitsSet,set.union(activeQubitsSet,currentQubitsSet),blockedQubitsSet,collectedGatesSoFar+[gateIndex])
        return res1+res2

    if len(set.intersection(relevantQubitsSet,currentQubitsSet))==len(currentQubitsSet) and len(set.intersection(activeQubitsSet,currentQubitsSet))>0:
        # This gate is fully inside the relevant set, i.e. no external dependency. And it does
        # use already active qubits. It can be taken. If we do not take it, then the touched qubits cannot be used later.
        #print("gate",gateIndex,"must be taken (or not and then block all further qubits)")
        res1=collectGates(qc,gateIndex+1,relevantQubitsSet,set.union(activeQubitsSet,currentQubitsSet),blockedQubitsSet,collectedGatesSoFar+[gateIndex])
        # Do not take it. Block qubits.
        blockedQubitsSetNew=set.union(blockedQubitsSet,set.intersection(currentQubitsSet,relevantQubitsSet))
        res2=collectGates(qc,gateIndex+1,relevantQubitsSet,activeQubitsSet,blockedQubitsSetNew,collectedGatesSoFar)
        return res1+res2
    print("strange gate. stop with index",gateIndex)

def getAllCandidates(qc, numQubits):
    quantumRegister=qc.qubits
    res=[]
    for tuple in itertools.combinations(list(range(len(quantumRegister))), numQubits):
        qubits=[]
        for t in tuple:
            qubits.append(quantumRegister[t])
        for i in range(len(qc)):
            #buffer=collectGates(qc, i, qubits)
            buffer=collectGates(qc,0,set(qubits),set([]),set([]),[])
            #print(buffer)
            #print(tuple, i, buffer)
            res=res+buffer
    res2=[]
    for r in res:
        if not(r in res2) and len(r)>0:
            res2.append(r)
    return res2

# Apply a reducer for a pattern to a quantum circuit.
def applyReducerPattern(qc,patternGateList,reducer):
    # Find the relevant qubits.
    qr=qc._qubits

    relevantGates=[]
    for c in patternGateList:
        relevantGates.append(qc[c])

    # Find the relevant qubits.
    relevantQubits=[]
    for g in relevantGates:
        relevantQubits=relevantQubits+g[1]
    relevantQubits=list(set(relevantQubits))
    relevantQubits=sortedQubits(qc,relevantQubits)
    relevantQubitsIndices=[]
    for q in relevantQubits:
        relevantQubitsIndices.append(qr.index(q))

    # Generate new circuit with replacements.
    qr2=QuantumRegister(len(qr))
    qc2=QuantumCircuit(qr2)
    for i in range(len(qc)):
        if not(i in patternGateList):
            newQubits=[]
            for q in qc[i][1]:
                newQubits.append(qr2[qr.index(q)])
            qc2.append(qc[i][0],newQubits)
        elif i==patternGateList[0]:
            insertPatternGates(qc2,qr2,relevantQubitsIndices,reducer)
    return qc2

def reduceCircuitByPattern(qc, consideredQubits, allPatterns, costPattern):
    allCandidateBlocks=getAllCandidates(qc,consideredQubits)

    candidateReducerTuples=[]
    for a in allCandidateBlocks:
        pattern=transformBlock(qc,a)
        reducers=reducePattern(pattern,allPatterns)
        for r in reducers:
            candidateReducerTuples.append((a,r))

    # Greedy: Take the best reduction first. Quality measured by a function.
    bestDiff=0
    newBestIndex=-1
    for iX in range(len(candidateReducerTuples)):
        x=candidateReducerTuples[iX]
        costOriginal=costPattern(transformBlock(qc,x[0]))
        costReplaced=costPattern(x[1])
        costReduction=costOriginal-costReplaced
        if not(costReduction==0):
            #print("reducer=",x,"reduces",transformBlock(qc,x[0]),"to",costReduction)
            pass
        if costReduction>bestDiff:
            bestDiff=costReduction
            newBestIndex=iX

    if bestDiff==0:
        print("best reduction of cost is 0")
        return qc

    bestReduction=candidateReducerTuples[newBestIndex]

    print("best cost reduction",bestDiff,"at index",newBestIndex,"out of",len(candidateReducerTuples),"possible reductions and it is",transformBlock(qc,bestReduction[0]),"=>",bestReduction[1])
    qc2=applyReducerPattern(qc,bestReduction[0],bestReduction[1])

    # Check the reduction.
    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    u=job.result().get_unitary()

    backend = Aer.get_backend('unitary_simulator')
    job2 = execute(qc2, backend)
    u2=job2.result().get_unitary()

    uDiff=u-u2
    isGood=True
    for i in range(len(uDiff)):
        for j in range(len(uDiff)):
            if abs(uDiff[i,j])>0.000001:
                isGood=False
    if not(isGood):
        print("WARNING: Reduction failed, unitaries are inconsistent!")
        print("apply failed",bestReduction,"on",transformBlock(qc,candidateReducerTuples[newBestIndex][0]))
    return qc2

def costPatternLength(pattern):
    return len(pattern)


