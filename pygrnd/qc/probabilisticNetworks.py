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

import random
import math
import numpy as np
from pygrnd.qc.helper import allCombinations, num2bin
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RYGate, ZGate

def monteCarloEvaluation(timesteps, nodes, probFail, probRecovery, edges, rounds=100000):
    """ Evaluate a probabilistic network with a Monte Carlo simulation. The output
        is a list of probabilities for all possible configurations.
    """
    allRes={}
    for run in range(rounds):
        # Initialization phase.
        res=[0]*len(nodes)
        for n in range(len(nodes)):
            node=nodes[n]
            if random.random()<probFail[node]:
                res[n]=1

        for t in range(timesteps-1):
            buffer=[0]*len(nodes)
            for n in range(len(nodes)):
                node=nodes[n]

                # Node was triggered in the last round. Now untrigger it with recovery probability.
                if res[n]==1:
                    if random.random()<probRecovery[node]:
                        buffer[n]=0
                    else:
                        buffer[n]=1
                else:
                    # The node was not triggered. Now consider intrinsic and trigger probability.

                    # intrinsic
                    if random.random()<probFail[node]:
                        buffer[n]=1

                    # Get ancestors.
                    ancestors=[]
                    for e in edges:
                        if e[1]==node:
                            ancestors.append(e[0])

                    # Triggered by ancestor.
                    for a in ancestors:
                        if res[nodes.index(a)]==1:
                            if random.random()<edges[(a,node)]:
                                buffer[n]=1
            # Prepare for next timestep.
            res=buffer

        # Collect result after all timesteps.
        tup=tuple(res)
        if tup in allRes:
            allRes[tup]=allRes[tup]+1
        else:
            allRes[tup]=1

    res2=[]
    for c in allCombinations(len(nodes)):
        tup=tuple([int(x) for x in list(c)])
        if tup in allRes:
            res2.append(allRes[tup])
        else:
            res2.append(0)
    return [x/rounds for x in res2]


def classicalEvaluation(timesteps, nodes, probFail, probRecovery, edges):
    """ Evaluate a probabilistic network with a classical calculation of the probabilities.
    """

    # Initialization in the first step. Just take the intrinsic probability for failure.

    # Storage format: original value, current value. We need to preserve the history
    # (just the previous step) to get the correct values of the ancestors in the
    # previous step and we have to keep track of them.

    res={'0'*len(nodes):1.0}

    for t in range(timesteps):

        buffer={}
        for a in allCombinations(len(nodes)):
            for b in allCombinations(len(nodes)):
                buffer[(a,b)]=0.0
        for r in res:
            buffer[(r,r)]=res[r]

        for n in range(len(nodes)):
            node=nodes[n]

            # This is just to collect the intermediate results without destroying the loop.
            buffer2={}
            for a in allCombinations(len(nodes)):
                for b in allCombinations(len(nodes)):
                    buffer2[(a,b)]=0.0

            for r in buffer:
                rLast=r[0]
                rNew=r[1]

                if rLast[n]=='1':
                    # Node was triggered in last time step. Now recover it with certain probability.
                    # It stays triggered.
                    buffer2[(rLast,rNew)]=buffer2[(rLast,rNew)]+buffer[r]*(1-probRecovery[node])
                    # It recovers.
                    rNew2 = rNew[:n] + '0' + rNew[n+1:]
                    buffer2[(rLast,rNew2)]=buffer2[(rLast,rNew2)]+buffer[r]*probRecovery[node]
                else:
                    # Node was not triggered in the last time step. Consider intrinsic and trigger probability.
                    # Gather ancestors.
                    ancestors=[]
                    for e in edges:
                        if e[1]==node:
                            ancestors.append(e[0])
                    # Calculate probability that it was not triggered.
                    probNotTriggered=1-probFail[node]
                    for a in ancestors:
                        if rLast[nodes.index(a)]=='1':
                            probNotTriggered=probNotTriggered*(1-edges[(a,node)])
                    # It stays off.
                    buffer2[(rLast,rNew)]=buffer2[(rLast,rNew)]+buffer[r]*probNotTriggered
                    # It gets triggered.
                    rNew2 = rNew[:n] + '1' + rNew[n+1:]
                    buffer2[(rLast,rNew2)]=buffer2[(rLast,rNew2)]+buffer[r]*(1-probNotTriggered)
            buffer=buffer2

        res={}
        for a in allCombinations(len(nodes)):
            res[a]=0.0
        for b in buffer:
            res[b[1]]=res[b[1]]+buffer[b]
    res2=[]
    for a in allCombinations(len(nodes)):
        if a in res:
            res2.append(res[a])
        else:
            res2.append(0.0)
    return res2

def hammingWeight(string):
    ''' Calculate the Hamming weight of a list with binary elements.
    '''
    if len(string)==0:
        return 0
    if string[0]=='0':
        return hammingWeight(string[1:])
    return hammingWeight(string[1:])+1

def gateOneTimestep(nodes, probFail, probRecovery, edges):
    ''' Construct a quantum gate that corresponds to one time step of
        the evolution of a probabilistic network.
    '''
    qr=QuantumRegister(2*len(nodes))
    qc=QuantumCircuit(qr)
    t=1
    for n in range(len(nodes)):
        node=nodes[n]

        # Treat recovery. If the node was 1 in the previous time step, then
        # turn it to 1 with 1-prob_recovery. And remove the rotation from the
        # intrinsic probability.
        #if probRecovery[node]>0:
        if True:
            angle=2*math.asin(math.sqrt(1-probRecovery[node]))-2*math.asin(math.sqrt(probFail[node]))
            qubitControl=(t-1)*len(nodes)+n
            qubitTarget=t*len(nodes)+n
            qc.append(RYGate(angle).control(),[qubitControl,qubitTarget])

        # Collect all nodes that influence the current node.
        ancestors=[]
        for e in edges:
            if e[1]==node:
                ancestors.append(e[0])

        # Gather qubits of ancestors in previous timestep.
        ancestorQubits=[]
        for a in ancestors:
            ancestorQubits.append((t-1)*len(nodes)+nodes.index(a))

        # Go through all combinations of all ancestor qubits that are not all zero.
        combis=allCombinations(len(ancestors))
        for c in combis:
            if hammingWeight(c)>0:

                # Calculate the probability that the node is not triggered under this combination.
                # All active ancestors do not trigger and the intrinsic is not triggered.
                # All binary combinations must be considered separately. Assume that
                # the node was not triggered in the previous time step (extra control qubits).
                probNotTriggered=1-probFail[node]
                for i in range(len(c)):
                    if c[i]=='1':
                        probNotTriggered=probNotTriggered*(1-edges[(ancestors[i],node)])
                angle=2*math.asin(math.sqrt(1-probNotTriggered))-2*math.asin(math.sqrt(probFail[node]))

                # Case with recovery probability of the node.
                qubitTarget=t*len(nodes)+n
                qubitTargetPrevious=(t-1)*len(nodes)+n

                # with recovery probability (if it was defaulted one step before, then it cannot be triggered again)
                control_state=c+'0'
                qc.append(RYGate(angle).control(num_ctrl_qubits=len(c)+1,ctrl_state=control_state[::-1]),ancestorQubits+[qubitTargetPrevious]+[qubitTarget])
    gate=qc.to_gate()
    gate.label='timestep'
    return gate

def createCircuit(timesteps, nodes, probFail, probRecovery, edges):
    """ Create the quantum circuit for a probabilistic network for the given
        number of time steps.
    """
    numberQubits=timesteps*len(nodes)
    qr=QuantumRegister(numberQubits)
    cr=ClassicalRegister(len(nodes),'c')
    qc=QuantumCircuit(qr,cr)

    gateTimestep=gateOneTimestep(nodes, probFail, probRecovery, edges)

    for t in range(timesteps):

        # First, set the intrinsic fail probability for each qubit in this time step.
        for n in range(len(nodes)):
            node=nodes[n]
            qubit=t*len(nodes)+n
            qc.ry(2*math.asin(math.sqrt(probFail[node])),qr[qubit])

        if t>0:
            qc.append(gateTimestep,qr[(t-1)*len(nodes):(t+1)*len(nodes)])

        qc.barrier()

    #qc.measure(qr[(timesteps-1)*len(nodes):],cr)
    return qc, qr, cr

def createCircuitNoBarrierNoClassicalBits(timesteps, nodes, probFail, probRecovery, edges):
    ''' A modification of the method createCircuit.
    '''
    numberQubits=timesteps*len(nodes)
    qr=QuantumRegister(numberQubits)
    qc=QuantumCircuit(qr)
    gateTimestep=gateOneTimestep(nodes, probFail, probRecovery, edges)
    for t in range(timesteps):

        # First, set the intrinsic fail probability for each qubit in this timestep.
        for n in range(len(nodes)):
            node=nodes[n]
            qubit=t*len(nodes)+n
            qc.ry(2*math.asin(math.sqrt(probFail[node])),qr[qubit])
        if t>0:
            qc.append(gateTimestep,qr[(t-1)*len(nodes):(t+1)*len(nodes)])

    return qc

def evaluateQuantum(timesteps, nodes, probFail, probRecovery, edges):
    """ Evaluate a probabilistic network with a simulation of the corresponding quantum circuit.
    """
    qc,qr,cr=createCircuit(timesteps, nodes, probFail, probRecovery, edges)
    backend_qiskit = Aer.get_backend(name='statevector_simulator')
    #job = execute(qc, backend_qiskit)
    #v = np.asarray(job.result().get_statevector())
    v = Statevector(qc)

    res={}
    for c in allCombinations(len(nodes)):
        res[c]=0

    for i in range(len(v)):
        binValue=num2bin(i,len(nodes)*timesteps)
        binValue2=binValue[:len(nodes)]
        res[binValue2]=res[binValue2]+abs(v[i])**2


    res2=[]
    for c in allCombinations(len(nodes)):
        d=c[::-1]
        if d in res:
            res2.append(res[d])
        else:
            res2.append(0)
    return res2

def constructGroverOperatorWithPhaseGateOnLastTimeStep(model, phaseGate, numberNodes):
    ''' Construct the Grover operator for the model gate of a probabilistic network
        model with time steps. The phase gate is applied to the part of the qubits
        that correspond to the last time step. The number of nodes of the
        network must be provided.
    '''
    numberQubits=model.num_qubits
    qr=QuantumRegister(numberQubits,'q')
    qc=QuantumCircuit(qr)

    # mark the good states sequentially
    qc.append(phaseGate,qr[-numberNodes:])

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

