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

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library.standard_gates import ZGate, XGate, UGate, PhaseGate
import math
import numpy as np
from pygrnd.qc.helper import num2bin

def list2string(lst):
    """ Helper function. Turn a list of integers into a binary string.
    """
    res=''
    for x in lst:
        if x==0:
            res='0'+res
        else:
            res='1'+res
    return res

def xor_sequence(a,b,n):
    """ Helper function.
         a index of first dimension
         b index of second dimension
         n number of qubits
    """
    ax=[int(x) for x in list(num2bin(a,n))]
    bx=[int(x) for x in list(num2bin(b,n))]
    seq=[]
    firstDiff=-1
    for i in range(n):
        if not ax[i]==bx[i]:
            firstDiff=i
            break
    for i in range(firstDiff+1,n):
        if not ax[i]==bx[i]:
            seq.append([firstDiff,i])
    return firstDiff,seq

def elim2block(i,j,a,b,n,qr,qc):
    """ Helper function. Elimination of a 2x2 block in a unitary matrix.
         i first index
         j second index
         a first component (complex)
         b second component (complex)
         n number qubits
    """
    ibin=[int(x) for x in list(num2bin(i,n))]
    jbin=[int(x) for x in list(num2bin(j,n))]
    first,seq = xor_sequence(i,j,n)
    norm=np.real(a*np.conj(a)+b*np.conj(b))
    anew=a/(math.sqrt(norm))
    bnew=b/(math.sqrt(norm))
    cnew=min(1.0,np.abs(np.conj(anew))) # avoid math domain errors from rounding problems
    theta=2*math.acos(cnew)
    phi=np.angle(-bnew)
    mu=np.angle(np.conj(anew))
    lam=np.angle(anew)-np.angle(-bnew)
    for s in seq:
        qc.cx(qr[n-s[0]-1],qr[n-s[1]-1])
    control=[]
    if ibin[first]==0:
        control=ibin
    else:
        control=jbin
    del control[first]
    ctrl_qubits=[]
    for i in reversed(range(0,n)):
        if i!=(n-first-1):
            ctrl_qubits.append(qr[i])
    ctrl_qubits.append(qr[n-first-1])
    qc.append(UGate(theta,phi,lam).control(num_ctrl_qubits=n-1,ctrl_state=list2string(control)),ctrl_qubits)
    qc.x(qr[n-first-1])
    qc.append(PhaseGate(mu).control(num_ctrl_qubits=n-1,ctrl_state=list2string(control)),ctrl_qubits)
    qc.x(qr[n-first-1])
    for s in reversed(seq):
        qc.cx(qr[n-s[0]-1],qr[n-s[1]-1])

def decomposer(u, qc, qr):
    """ Generate a circuit that implements the unitary on a register (at least 2 qubits).
    """
    n=len(qr)
    u2=u.copy()
    u2=np.linalg.inv(u2)
    A0=np.identity(2**n,dtype=complex)
    for i in range(len(u2)):
        for j in range(i+1,len(u2)):
            a=u2[i][i]+0j
            b=u2[j][i]+0j
            norm=a*np.conj(a)+b*np.conj(b)
            if abs(norm)>0:
                norm=norm.real
                an=a/math.sqrt(norm)
                bn=b/math.sqrt(norm)
                A=np.identity(2**n,dtype=complex)
                A[i][i]=np.conj(an)
                A[i][j]=np.conj(bn)
                A[j][i]=-bn
                A[j][j]=an
                elim2block(i,j,a,b,n,qr,qc)
                u2=np.matmul(A,u2)
                A0=np.matmul(A,A0)
    A=np.identity(2**n,dtype=complex)
    lastPos=len(u2)-1
    A[lastPos][lastPos]=np.conj(u2[lastPos][lastPos])
    u2=np.matmul(A,u2)
    A0=np.matmul(A,A0)
    qc.append(PhaseGate(np.angle(A[lastPos][lastPos])).control(num_ctrl_qubits=n-1,ctrl_state='1'*(n-1)),reversed(qr))

def circuitStateVector(v, qc, qr):
    """ Generate the inverse of a circuit that creates a state v on a register with (at least 2 qubits).
    """
    n=len(qr)
    u2=np.zeros((len(v),len(v)),dtype=complex)
    for i in range(len(v)):
        u2[i][0]=v[i]
    A0=np.identity(2**n,dtype=complex)
    for i in range(1):
        for j in range(i+1,len(u2)):
            a=u2[i][i]+0j
            b=u2[j][i]+0j
            norm=a*np.conj(a)+b*np.conj(b)
            if abs(norm)>0:
                norm=norm.real
                an=a/math.sqrt(norm)
                bn=b/math.sqrt(norm)
                A=np.identity(2**n,dtype=complex)
                A[i][i]=np.conj(an)
                A[i][j]=np.conj(bn)
                A[j][i]=-bn
                A[j][j]=an
                elim2block(i,j,a,b,n,qr,qc)
                u2=np.matmul(A,u2)
                A0=np.matmul(A,A0)
    A=np.identity(2**n,dtype=complex)
    lastPos=len(u2)-1
    A[lastPos][lastPos]=np.conj(u2[lastPos][lastPos])
    u2=np.matmul(A,u2)
    A0=np.matmul(A,A0)
    qc.append(PhaseGate(np.angle(A[lastPos][lastPos])).control(num_ctrl_qubits=n-1,ctrl_state='1'*(n-1)),reversed(qr))

def controlledXGate(controls, ancilla, target, qc):
    """ Implement an X gate, which is controlled by multiple qubits, with Toffoli gates and
        one ancilla qubit. This method creates an exponential number of Toffoli gates in
        the number of control qubits.
    """
    if len(controls)>3:
        controlledXGate(controls[:-1],controls[-1],ancilla,qc)
    else:
        qc.append(XGate().control(len(controls)-1),controls[:-1]+[ancilla])

    qc.ccx(controls[-1],ancilla,target)

    if len(controls)>3:
        controlledXGate(controls[:-1],controls[-1],ancilla,qc)
    else:
        qc.append(XGate().control(len(controls)-1),controls[:-1]+[ancilla])
    qc.ccx(controls[-1],ancilla,target)

def controlledXGateManyAncillas(controls, ancillas, target, qc):
    """ Helper function. This is an implementation of lemma 7.2 of Barenco et. al.,
        Elementary gates for quantum computation, Physical Review A52,
        3457 (1995).
    """
    if len(controls)==2:
        qc.ccx(controls[0],controls[1],target)
        return
    cbits=len(controls)
    allAncillas=ancillas+[target]
    #print(allAncillas)
    for i in range(cbits-2):
        qc.ccx(controls[-i-1],allAncillas[-i-2],allAncillas[-i-1])
    qc.ccx(controls[0],controls[1],allAncillas[-(cbits-3)-2])
    for i in range(cbits-2)[::-1]:
        qc.ccx(controls[-i-1],allAncillas[-i-2],allAncillas[-i-1])
    for i in range(1,cbits-2):
        qc.ccx(controls[-i-1],allAncillas[-i-2],allAncillas[-i-1])
    qc.ccx(controls[0],controls[1],allAncillas[-(cbits-3)-2])
    for i in range(1,cbits-2)[::-1]:
        qc.ccx(controls[-i-1],allAncillas[-i-2],allAncillas[-i-1])

def controlledXGateToffoliDecomposition(controls, ancilla, target, qc):
    """ Implement an X gate with many controls with a linear number of Toffoli gates using one ancilla. This
        is an implementation of corollary 7.4 of Barenco et. al., Elementary gates for quantum computation,
        Physical Review A52, 3457 (1995).
    """
    m1=math.ceil(len(controls)/2)
    m2=len(controls)-m1

    if m1>2:
        controlledXGateManyAncillas(controls[:m1], controls[m1:], ancilla, qc)
    else:
        qc.append(XGate().control(m1),controls[:m1]+[ancilla])

    if m2+1>2:
        controlledXGateManyAncillas(controls[m1:]+[ancilla], controls[:m1], target, qc)
    else:
        qc.append(XGate().control(m2+1),controls[m1:m1+m2]+[ancilla]+[target])

    if m1>2:
        controlledXGateManyAncillas(controls[:m1], controls[m1:m1+m2], ancilla, qc)
    else:
        qc.append(XGate().control(m1),controls[:m1]+[ancilla])

    if m2+1>2:
        controlledXGateManyAncillas(controls[m1:]+[ancilla], controls[:m1], target, qc)
    else:
        qc.append(XGate().control(m2+1),controls[m1:]+[ancilla]+[target])
