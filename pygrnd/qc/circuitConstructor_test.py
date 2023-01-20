'''Copyright 2023 JoS QUANTUM GmbH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from pygrnd.qc.circuitConstructor import *
from pygrnd.qc.helper import allCombinations
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute
from itertools import permutations
import random

def getDiffRandomUnitary(num_qubits):
    """ Generate a random circuit on several qubits and get the corresponding unitary.
        Decompose the unitary into uncontrolled and controlled operations on qubits
        and get the unitary for this circuit. Return the norm of the difference to
        check the correctness of the decomposition.
    """
    qc=random_circuit(num_qubits, depth=random.choice(range(10,20)))
    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    uOriginal = np.asarray(job.result().get_unitary())

    qr=QuantumRegister(num_qubits)
    qc=QuantumCircuit(qr)
    decomposer(uOriginal, qc, qr)

    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    uConstruct = np.asarray(job.result().get_unitary())

    return np.linalg.norm(uOriginal-uConstruct)

#
# Generate many random circuits and compare unitaries with getDiffRandomUnitary.
#
# biggestError=0
# for num_qubits in range(2,5):
#     for i in range(100):
#         deviation=getDiffRandomUnitary(num_qubits)
#         if deviation>biggestError:
#             biggestError=deviation
#             print("new biggest error for num_qubits/i/error",num_qubits,i,biggestError)


def getDiffRandomStatePrep(qubits):
    """ Create a random state (including complex phases) and calculate
        the difference of the given statevector with the statevector
        from the simulator for the circuit that generates the state.
    """

    n=2**qubits

    v=[random.random()+1j*random.random() for x in range(n)]
    norm=0
    for x in v:
        norm=norm+abs(x)**2
    v2=[x/math.sqrt(norm) for x in v]

    qr=QuantumRegister(qubits,'q')
    qc=QuantumCircuit(qr)
    circuitStateVector(v2, qc, qr)
    gate=qc.to_gate()

    qr2=QuantumRegister(qubits,'q')
    qc2=QuantumCircuit(qr2)
    qc2.append(gate.inverse(),qr2)
    backend=Aer.get_backend('statevector_simulator')
    job=execute(qc2,backend)
    v3=np.asarray(job.result().get_statevector())

    diff=0
    for i in range(n):
        diff=diff+abs(v2[i]-v3[i])
    return diff

#
# Run many random tests for state preparation on 2-4 qubits.
#
# for qubits in range(2,5):
#     for i in range(100):
#         value=getDiffRandomStatePrep(qubits)
#         if abs(value)>0.00000000001:
#             print("error!")
#     print("qubits done: ", qubits)

#
# Test permutation matrices
#
# qubits=3
#
# biggestError=0
# perm = permutations(list(range(2**qubits)))
# for p in perm:
#     m=np.zeros((2**qubits,2**qubits))
#     for i in range(2**qubits):
#         m[i][p[i]]=1
#     qr=QuantumRegister(qubits)
#     qc=QuantumCircuit(qr)
#     decomposer(m, qc, qr)
#
#     backend = Aer.get_backend('unitary_simulator')
#     job = execute(qc, backend)
#     uConstruct = np.asarray(job.result().get_unitary())
#     error=np.linalg.norm(m-uConstruct)
#     if error>biggestError:
#         biggestError=error
#         print("new biggest error for perm/error:",p,error)

#
# Check states that come from binary strings.
#
# qubits=4
# combi=allCombinations(2**qubits)
# biggestError=0
# for c in combi:
#     v=[int(x) for x in c]
#     norm=0
#     for x in v:
#         norm=norm+abs(x)
#     if norm>0:
#         v2=[x/math.sqrt(norm) for x in v]
#         qr=QuantumRegister(qubits,'q')
#         qc=QuantumCircuit(qr)
#         circuitStateVector(v2, qc, qr)
#         gate=qc.to_gate()
#
#         qr2=QuantumRegister(qubits,'q')
#         qc2=QuantumCircuit(qr2)
#         qc2.append(gate.inverse(),qr2)
#         backend=Aer.get_backend('statevector_simulator')
#         job=execute(qc2,backend)
#         v3=np.asarray(job.result().get_statevector())
#         error=np.linalg.norm(v2-v3)
#
#         if error>biggestError:
#             biggestError=error
#             print("new biggest error:",biggestError)

#
# Test all permutations of the qubits for the simple decomposition with
# exponential overhead.
#
# qubits=5
#
# for perm in permutations(range(qubits)):
#     qr=QuantumRegister(qubits,'q')
#     qc=QuantumCircuit(qr)
#
#     qr2=[qr[perm[x]] for x in range(qubits)]
#
#     qc.append(XGate().control(qubits-2),qr2[:qubits-2]+[qr2[qubits-1]])
#     qc.barrier()
#     controlledXGate(qr2[:qubits-2], qr2[qubits-2], qr2[qubits-1], qc)
#
#     backend_sim = Aer.get_backend('unitary_simulator')
#     job_sim = execute(qc, backend_sim)
#     u=np.asarray(job_sim.result().get_unitary())
#     error=np.linalg.norm(u-np.identity(2**qubits))
#     if error>0.00000001:
#         print("perm/error:",perm,error)

#
# Test all permutations of the qubits.
#
# qubits=6
#
# for perm in permutations(range(qubits)):
#     qr=QuantumRegister(qubits,'q')
#     qc=QuantumCircuit(qr)
#
#     qr2=[qr[perm[x]] for x in range(qubits)]
#
#     qc.append(XGate().control(qubits-2),qr2[:qubits-2]+[qr2[qubits-1]])
#     qc.barrier()
#     controlledXGateSplitStarter(qr2[:qubits-2], qr2[qubits-2], qr2[qubits-1], qc)
#     if random.random()<0.001:
#         display(qc.draw(output='mpl'))
#
#     backend_sim = Aer.get_backend('unitary_simulator')
#     job_sim = execute(qc, backend_sim)
#     u=np.asarray(job_sim.result().get_unitary())
#     error=np.linalg.norm(u-np.identity(2**qubits))
#     if error>0.00000001:
#         print("perm/error:",perm,error)
