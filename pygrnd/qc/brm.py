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

## import qiskit to build circuits
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import U3Gate
from qiskit import execute
from qiskit import Aer




def brm(RIlist, TPlist, model2gate=False):

    ## input:
    #  Risk item list e.g.  RIlist = ["p0=0.1","p1=0.2","p2=0.3"]
    #  Transition risk e.g. TPlist = ["0->1=0.2","0->2=0.3"]
    # output: either circuit (model2gate=False) OR gate (model2gate=True)

    qr=QuantumRegister(len(RIlist),'q')
    circuitname = QuantumCircuit(qr)
    # now find RI which cannot be triggered by transitions and put uncontrolled u3 gates in for them
    
    
    qr=QuantumRegister(len(RIlist),'q')
    circuitname = QuantumCircuit(qr)
    RI=[]  # list of probabilities for risk items
    cv=[]  # control vector: for each column of the matrix of probabilities (RI on diagonal), all the ones ending in a given RI)
    for i in RIlist:
        q=str(i)
        res = q.partition("=")[2]
        RI.append(float(res))
    mat = np.zeros((len(RI),len(RI)))
    for i in range(len(RIlist)):
        mat[i,i] = RI[i]
    for i in TPlist:
        q=str(i)
        res = q.partition("->")[0]
        x=int(res)
        res = q.partition("->")[2]
        y=int(res.partition("=")[0])
        p=float(res.partition("=")[2])
        mat[x,y] = p
    
    
    for x in range(len(RIlist)):
        cx=0
        cv=[]
    #       print("checing col",x)
        for y in range(len(RIlist)):
            if mat[y,x] !=0 and x>y:
                cx=1
                cv.append(y)
        if cx==0:
    #            print("# not controlled:",x)
            #print("test",x)
            #print(2*math.asin(math.sqrt(mat[x,x])))
            circuitname.u(2*math.asin(math.sqrt(mat[x,x])),0,0,qr[x])
        else:
            if len(cv)>1:                                                           # this RI is triggered by more than one other RI. The triggering RI are in the list "cv"
                print("NOTE: Item",x,"is triggered by more than one other RI!")
                #print(cv)
                controllist=[]
                for i in range(len(cv)):
                    controllist.append(qr[cv[i]])
                controllist.append(qr[y])
                for i in range(2**len(cv)):
                    cts = format(i, "0"+str(len(cv))+"b")
                    print(cts)
                    if i==0:
                        p = mat[y,y]
                    else:
                        p=1
                        pbef=0
                        print("ITEM:")
                        for j in range(len(cv)):
                            if cts[j]=="1":
                                print("mat[y,j]",mat[j,y])
                                p=p*(1-pbef)*mat[j,y]
                                pbef=mat[j,y]

                    #print("Probability",p)
                    #print(controllist)
                    circuitname.append(U3Gate(2*math.asin(math.sqrt(p)),0,0).control(num_ctrl_qubits=len(cv),ctrl_state=cts),controllist)
    # Here we can insert a loop that goes through all 2**len(cv) combinations of possibilities to control the RI and calculate the probability and put in a multiply controlled U3-gate 
            if len(cv)==0:
                print("there's an empty risk item ...???")
            else:
    #                print(mat[x,x])
                circuitname.append(U3Gate(2*math.asin(math.sqrt(mat[x,x])),0,0).control(num_ctrl_qubits=1,ctrl_state='0'),[qr[cv[0]],qr[x]])
                ptrig = mat[cv[0],x] + (1-mat[cv[0],x])*mat[x,x]
                circuitname.append(U3Gate(2*math.asin(math.sqrt(ptrig)),0,0).control(num_ctrl_qubits=1,ctrl_state='1'),[qr[cv[0]],qr[x]])
    
    if model2gate==True:
        gate=circuitname.to_gate()
        gate.label="BRM"
        return gate, mat
    
    if model2gate==False:
        #gate=circuitname.to_gate()
        #gate.label="BRM"
        return circuitname, mat
    
    #return circuitname, gate #.draw(output='mpl')
