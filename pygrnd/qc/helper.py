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

import numpy as np
import math
from qiskit.circuit.library import XGate

def num2bin(x,r):
    res=""
    buffer=x
    for i in range(r):
        m=buffer%2
        res=res+str(m)
        buffer=(buffer-m)//2
    return res[::-1]

def bin2num(c):
    if len(c)==0:
        return 0
    elif c[-1]=='0':
        return 2*bin2num(c[:-1])
    else:
        return 2*bin2num(c[:-1])+1

def allCombinations(n):
    if n==0:
        return ['']
    res=[]
    for a in allCombinations(n-1):
        res.append(a+'0')
        res.append(a+'1')
    return res

def counts2probs(counts):
    bits=len(list(counts.keys())[0])
    summe=0
    for a in allCombinations(bits):
        if a in counts:
            summe=summe+counts[a]
    res=[]
    for a in allCombinations(bits):
        if a in counts:
            res.append(counts[a]/summe)
        else:
            res.append(0.0)
    return res

def maxString(counts):
    bits=len(list(counts.keys())[0])
    probList=counts2probs(counts)
    stringList=allCombinations(bits)
    maxi=np.argmax(probList)
    return stringList[maxi]
    
def showQAEoutput(counts,STATELIST,QAEqubits):
    print("Bin with the highest probability: ",maxString(counts))
    print("Number of Bin with the highest probability: ",bin2num(maxString(counts)))
    probTail=math.sin(bin2num(maxString(counts))*math.pi/(2**QAEqubits))**2
    print("The probability of the tail event ",STATELIST," is: ",probTail)
    return probTail

def addPower2(qr, qc, power, qubits):
    """ Add the gates on the register that correspond to the addition of 2^power.
    """
    for i in range(power,qubits)[::-1]:
        controls=[qr[j] for j in range(power,i+1)]
        if len(controls)==1:
            qc.x(controls)
        else:
            qc.append(XGate().control(num_ctrl_qubits=len(controls)-1),controls)

def addValue(qr,qc,value):
    """ Add the gates on the register that correspond to the addition of value.
    """
    bits=num2bin(value,len(qr))
    power=0
    for x in bits[::-1]:
        if x=='1':
            addPower2(qr, qc, power, len(qr))
        power=power+1

def subtractPower2(qr, qc, power, qubits):
    """ Add the gates on the register that correspond to the subtraction of 2^power.
    """
    for i in range(power,qubits):
        controls=[qr[j] for j in range(power,i+1)]
        if len(controls)==1:
            qc.x(controls)
        else:
            qc.append(XGate().control(num_ctrl_qubits=len(controls)-1),controls)

def subtractValue(qr,qc,value):
    """ Add the gates on the register that correspond to the subtraction of
        the specified value.
    """
    bits=num2bin(value,len(qr))
    power=0
    for x in bits[::-1]:
        if x=='1':
            subtractPower2(qr, qc, power, len(qr))
        power=power+1

