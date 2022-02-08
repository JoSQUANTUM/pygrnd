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

import itertools
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RGate,U3Gate,GMS,QFT,XGate,RXXGate,ZGate,PhaseGate,SwapGate,CXGate,RXGate,RYGate,RZGate, SXGate, SXdgGate, CPhaseGate, HGate,CCXGate
from qiskit import Aer
from qiskit import execute


def decodeParameter(code):
    phaseA=0.1
    phaseB=0.6
    basisParameters=[phaseA,phaseB]
    if type(code)==type([]):
        res=0
        for i in range(len(code)):
            res=res+code[i]*basisParameters[i]
        return res
    elif type(code)==type(1.1):
        return code
    else:
        print("Error: Unsupported phase value (no list no double)")

def insertPatternGates(qc, qr, relevantQubitsIndices, a):
    for gateInfo in a:
        currentGate=gateInfo[0]
        currentParams=gateInfo[1]
        currentQubits=gateInfo[2]
        currentRegisterQubits=[]

        for c in currentQubits:
            relevantQubit=relevantQubitsIndices[c]
            currentRegisterQubits.append(qr[relevantQubit])
        if currentGate=="X":
            qc.x(currentRegisterQubits)
        elif currentGate=="CNOT":
            qc.cx(currentRegisterQubits[0],currentRegisterQubits[1])
        elif currentGate=="CCX":
            qc.ccx(currentRegisterQubits[0],currentRegisterQubits[1],currentRegisterQubits[2])
        elif currentGate=="CZ":
            qc.cz(currentRegisterQubits[0],currentRegisterQubits[1])
        elif currentGate=="SWAP":
            qc.swap(currentRegisterQubits[0],currentRegisterQubits[1])
        elif currentGate=="MS":
            qc.append(GMS(2,[[0,math.pi/2],[0,0]]),[currentRegisterQubits[0],currentRegisterQubits[1]])
        elif currentGate=="P":
            valueLambda=decodeParameter(currentParams['lambda'])
            qc.append(PhaseGate(valueLambda),currentRegisterQubits)
        elif currentGate=="U3":
            valueTheta=decodeParameter(currentParams['theta'])
            valuePhi=decodeParameter(currentParams['phi'])
            valueLambda=decodeParameter(currentParams['lambda'])
            qc.append(U3Gate(valueTheta,valuePhi,valueLambda),currentRegisterQubits)
        elif currentGate=="R":
            valueTheta=decodeParameter(currentParams['theta'])
            valuePhi=decodeParameter(currentParams['phi'])
            qc.append(RGate(valueTheta,valuePhi),currentRegisterQubits)
        elif currentGate=="RXX":
            valueTheta=decodeParameter(currentParams['theta'])
            qc.append(RXXGate(valueTheta),currentRegisterQubits)
        elif currentGate=="GPI":
            valuePhi=decodeParameter(currentParams['phi'])
            qc.append(U3Gate(math.pi,valuePhi,-valuePhi+math.pi),currentRegisterQubits)
        elif currentGate=="GPI2":
            valuePhi=decodeParameter(currentParams['phi'])
            qc.append(RGate(math.pi/2,valuePhi),currentRegisterQubits)
        elif currentGate=="GZ":
            valueTheta=decodeParameter(currentParams['theta'])
            qc.append(RZGate(valueTheta),currentRegisterQubits)
        elif currentGate=="Z":
            qc.z(currentRegisterQubits)
        elif currentGate=="S":
            qc.s(currentRegisterQubits)
        elif currentGate=="Sdg":
            qc.sdg(currentRegisterQubits)
        elif currentGate=="T":
            qc.t(currentRegisterQubits)
        elif currentGate=="Tdg":
            qc.tdg(currentRegisterQubits)            
        elif currentGate=="H":
            qc.h(currentRegisterQubits)
        elif currentGate=="SX":
            qc.sx(currentRegisterQubits)
        elif currentGate=="SXdg":
            qc.sxdg(currentRegisterQubits)
        elif currentGate=="CP":
            valueLambda=decodeParameter(currentParams['lambda'])
            qc.append(PhaseGate(valueLambda).control(1),currentRegisterQubits)
        elif currentGate=="RX":
            valueTheta=decodeParameter(currentParams['theta'])
            qc.append(RXGate(valueTheta),currentRegisterQubits)
        elif currentGate=="RY":
            valueTheta=decodeParameter(currentParams['theta'])
            qc.append(RYGate(valueTheta),currentRegisterQubits)
        elif currentGate=="RZ":
            valueLambda=decodeParameter(currentParams['lambda'])
            qc.append(RZGate(valueLambda),currentRegisterQubits)
        else:
            print("UNKNOWN gate:",currentGate)

def allPossibleCircuits(totalQubits, gates, params, qubits, gateNumber):
    if gateNumber==0:
        return [[]]
    recursiveSolutions=allPossibleCircuits(totalQubits, gates, params, qubits, gateNumber-1)

    res=[]
    for i in range(len(gates)):
        for p in itertools.combinations(list(range(totalQubits)),qubits[i]):
            for t in itertools.permutations(p):
                for r in recursiveSolutions:
                    res.append(r+[[gates[i],params[i],list(t)]])
    return res

def getIdentities(totalQubits, numberGates, prefixPattern, gates, params, qubits):
    all=allPossibleCircuits(totalQubits, gates, params, qubits, numberGates)
    print("Search space size:",len(all))
    good=[]
    counter=0
    for a in all:
        counter=counter+1
        if (counter%1000000)==0:
            print("searched already:",round(100*counter/len(all),2), "% found so far:",len(good))

        # Create circuit, possibly with prefix.
        a=prefixPattern+a
        qr=QuantumRegister(totalQubits)
        qc=QuantumCircuit(qr)
        insertPatternGates(qc,qr,list(range(totalQubits)),a)

        backend = Aer.get_backend('unitary_simulator')
        job = execute(qc, backend)
        u=job.result().get_unitary()
        isId=True
        for i in range(len(u)):
            for j in range(len(u)):
                v=u[i,j]
                if i==j and abs(v-1)>0.00001:
                    isId=False
                if not(i==j) and abs(v)>0.00001:
                    isId=False
        if isId:
            good.append(a)
    return good
