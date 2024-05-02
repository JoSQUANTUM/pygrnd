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
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import ZGate
from qiskit.providers.basic_provider import BasicProvider
from qiskit import transpile


def brmoracle(name,PDFgenerator,pdfqubits,pdfancillas,LISTOFcontrolstrings):

   ##input:
   # PDFgenerator = underlying risk model
   # pdfqubits = QAE bit resolution
   # LISTOFcontrolstrings = string of states that we are searching the overall probability 

    
    q=QuantumRegister(pdfqubits+pdfancillas+1,'q')
    circ=QuantumCircuit(q)

    circ.x(q[pdfqubits+pdfancillas])   # the last one in the list is used as indicatorbit
    # reflection about target states
    for i in LISTOFcontrolstrings:
        controlstring=i
        circ.append(ZGate().control(num_ctrl_qubits=len(i),ctrl_state=i),q[pdfancillas:pdfqubits+pdfancillas+1])
    circ.x(q[pdfqubits+pdfancillas])   # the last one in the list is used as indicatorbit
    
    #inverse operation
    circ.append(PDFgenerator.inverse(),q[0:pdfqubits+pdfancillas])

    # reflection about 0
    circ.x(q[pdfqubits+pdfancillas])   # the last one in the list is used as indicatorbit
    z=""
    circ.append(ZGate().control(num_ctrl_qubits= len(LISTOFcontrolstrings[0]) ,ctrl_state=z.zfill(len(LISTOFcontrolstrings[0]))),q[pdfancillas:pdfqubits+pdfancillas+1])
    circ.x(q[pdfqubits+pdfancillas])   # the last one in the list is used as indicatorbit

    # operation
    circ.append(PDFgenerator,q[0:pdfqubits+pdfancillas])

    # overall -1
    circ.x(q[pdfqubits+pdfancillas-1])
    circ.z(q[pdfqubits+pdfancillas-1])
    circ.x(q[pdfqubits+pdfancillas-1])
    circ.z(q[pdfqubits+pdfancillas-1])


    return circ
