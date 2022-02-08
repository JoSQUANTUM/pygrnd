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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates import ZGate
from qiskit.circuit.library import QFT


def qae(QAEqubits, inqubits, modelinqubits, A, Q, qae2gate=False):

# outqubits is the number of qubits to use for the output
# inqubits 
# modelinqubits is the number of qubits A requires (this may include ancillas, ie is >= inqubits)
# A is a gate that generates the input to be estimated
# Q is the oracle (one qubit larger than A and controlled)

    q=QuantumRegister(modelinqubits+QAEqubits+1,'q')
    c=ClassicalRegister(QAEqubits,'c')
    circuitname=QuantumCircuit(q,c)

    for i in range(QAEqubits):
        circuitname.h(q[i])

    circuitname.append(A,q[QAEqubits:QAEqubits+modelinqubits])

    pwr=1
    for i in range(QAEqubits):
        circuitname.append(Q.power(pwr).control(1),[q[QAEqubits-i-1],*q[QAEqubits:QAEqubits+modelinqubits+2]])
        #circuitname.append(Q.power(pwr),[q[outqubits-i-1],*q[outqubits:outqubits+modelinqubits+2]])
        pwr=pwr*2

    circuitname.append(QFT(QAEqubits,do_swaps=False).inverse(),q[:QAEqubits])
    circuitname.measure(q[0:QAEqubits],c[0:QAEqubits])
    
    if qae2gate==True:
        gate=circuitname.to_gate()
        gate.label="BRM"
        return gate, mat
    
    if qae2gate==False:
        #gate=circuitname.to_gate()
        #gate.label="BRM"
        return circuitname

