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

# this can be used to formulate the Unit Comittment Problem as a Quadratic Unconstrained Binary Optimization (QUBO) matrix
# currently power units can be 0 or 1 only, resolution will be introduced later with a logarithmic overhead using binary encoding


import numpy as np
import dimod
import greedy
import pandas as pd


#
# Helper: Create all runs of mini elements.
#
def createValidRuns(mini, n):
    res=[]
    for i in range(n-mini+1):
        buffer=[]
        for j in range(mini):
            buffer.append(i+j+1)
        res.append(buffer)
    return res



# #### Conditions for Minimum Up time
# - activation variables $a_1, \ldots, a_n$ for a single power station
# - ancilla variable $a_{n+1}$ means that first time slice of length mini is active, $a_{n+2}$ is second time slice, etc.
# - number of ancilla qubits depends on mini and n
# - setA equations: $a_{n+1}\rightarrow a_i$ for all suitable activation variables $a_i$
# - setB equations: $a_i \rightarrow a_{n+k} \lor \ldots \lor a_{n+\ell}$ for all suitable ancilla variables $a_{n+k},\ldots, a_{n+\ell}$

# #### Conditions for Minimum Down time
# - ancilla variable $a_{n+1}$ means that first time slice of length mini is inactive, $a_{n+2}$ is second time slice, etc.
# - number of ancilla qubits depends on mini and n
# - we add an offset to take into account the ancillas from setA and setB for minimum up time (considered first)
# - setC equations: $a_{n+1}\rightarrow \neg a_i$ for all suitable activation variables $a_i$
# - setD equations: $\neg a_i \rightarrow a_{n+k} \lor \ldots \lor a_{n+\ell}$ for all suitable ancilla variables $a_{n+k},\ldots, a_{n+\ell}$


#
# Calculate the number of ancillas.
#
def numberAncillas(mini,n):
    return len(createValidRuns(mini,n))

def setA(mini, n):
    possible=createValidRuns(mini,n)
    res=[]
    for i in range(len(possible)):
        for z in possible[i]:
            res.append([-(n+i+1),z])
    return res

def setB(mini,n):
    possible=createValidRuns(mini,n)
    res=[]
    for i in range(1,n+1):
        buffer=[-i]
        for j in range(len(possible)):
            if i in possible[j]:
                buffer.append((n+j+1))
        res.append(buffer)
    return res



#
# Create "ancilla_i -> a_j" for all suitable j
#
# skipAncilla denotes the number of ancillas of setA and setB and they must be skipped.
# The number can be determined by numberAncillas(mini,n).
#
def setC(mini, n, skipAncillas):
    possible=createValidRuns(mini,n)
    res=[]
    for i in range(len(possible)):
        for z in possible[i]:
            res.append([-(skipAncillas+n+i+1),-z])
    return res


#
# Create "a_i -> ancilla_a or ... or ancilla_b" for all suitable ancillas.
#
def setD(mini,n,skipAncillas):
    possible=createValidRuns(mini,n)
    #print(possible)
    res=[]
    for i in range(1,n+1):
        buffer=[i]
        for j in range(len(possible)):
            if i in possible[j]:
                buffer.append((skipAncillas+n+j+1))
        res.append(buffer)
    return res


# Just return not(a1 and a2 and a3) and all shifted versions of it
def setE(maxi,n):
    possible=createValidRuns(maxi+1,n)
    res=[]
    for p in possible:
        buffer=[]
        for c in p:
            buffer.append(-c)
        res.append(buffer)
    return res

# Just return not(a1 and a2 and a3) and all shifted versions of it
def setF(maxi,n):
    possible=createValidRuns(maxi+1,n)
    return possible





# Combine types of equations
# NOT NEEDED CURRENTLY
def getEquations(mini,n,numberAncillas):
    buffer=setA(mini,n)+setB(mini,n)+setC(mini,n,numberAncillas)+setD(mini,n,numberAncillas)
    return buffer




def allBinaryStrings(n):
    if n==0:
        return ['']
    else:
        buf=[]
        for a in allBinaryStrings(n-1):
            buf.append(a+'0')
            buf.append(a+'1')
    return buf


def convertStringToAssignment(string):
    res=[]
    for s in string:
        if s=='0':
            res.append(False)
        else:
            res.append(True)
    return res


def evalClause(clause, assignment):
    if len(clause)==0:
        return True
    result=False
    for c in clause:
        if c>0:
            if assignment[c-1]:
                result=True
        if c<0:
            if not(assignment[-c-1]):
                result=True
    return result


def evalFormula(formula, assignment):
    res=True
    for clause in formula:
        if not(evalClause(clause,assignment)):
            res=False
    return res





#
# n is number of steps. SAT formulas for 1 unit only!
#
def generateSAT(minDown, maxDown, minUp, maxUp, n):
    res=[]
    minUpAncillas=0
    minDownAncillas=0
    if minUp>0:
        res=res+setA(minUp,n)+setB(minUp,n)
        minUpAncillas=numberAncillas(minUp,n)
    if minDown>0:
        res=res+setC(minDown,n,minUpAncillas)+setD(minDown,n,minUpAncillas)
        minDownAncillas=numberAncillas(minDown,n)
    if maxUp>0:
        res=res+setE(maxUp,n)
    if maxDown>0:
        res=res+setF(maxDown,n)
    #return res,minUpAncillas+minDownAncillas
    return res


# Concatenate equations for two power stations. Shift everything of second power station.
#
def concat3SAT(formulasA,formulasB):
    res=[]
    maxIndex=0
    for formula in formulasA:
        buffer=[]
        for x in formula:
            maxIndex=max(maxIndex,abs(x))
            buffer.append(x)
        res.append(buffer)
    for formula in formulasB:
        buffer=[]
        for x in formula:
            if x>0:
                buffer.append(x+maxIndex)
            else:
                buffer.append(x-maxIndex)
        res.append(buffer)
    return res,maxIndex


# Concatenate equations for several power stations. List of SAT for each power station is input.
# Return collected formulas and mappings.
#
def concat3SATMany(formulas,shift,n):
    
    #print("concat with formulas=",formulas)
    
    if len(formulas)==0:
        return [],[] # new formulas/mappings

    # Treat the first power station with the shift and find the number of variables for this power station.
    currentRes=[]
    currentMax=0
    for currentFormula in formulas[0]:
        buffer=[]
        for x in currentFormula:
            currentMax=max(currentMax,abs(x))
            if x>0:
                buffer.append(x+shift)
            else:
                buffer.append(x-shift)
        currentRes.append(buffer)
        recFormulas,recMappings=concat3SATMany(formulas[1:],shift+currentMax,n)
        #print("recF",recFormulas)
        #print("recM",recMappings)
    return [currentRes]+recFormulas,[list(range(shift,shift+n))]+recMappings


# The wrapper for concatenating the formulas for the min up times.
#
def concat3SATManyWrapper(formulas,n):
    bufferFormulas,bufferMappings=concat3SATMany(formulas,0,n)
    res=[]
    for b in bufferFormulas:
        for x in b:
            res.append(x)
    return res,bufferMappings


# optimize concatSATMany for performance through iterations

def concat3SATManyIterative(formulas,shift,n):

    #print("concat with formulas=",formulas)
    #print("concat with formulas=",formulas.reverse())

    finalRes=[]
    finalMapping=[]
    shift=0
    currentMax=0

    for f in formulas:

    # Treat the first power station with the shift and find the number of variables for this power station
        currentRes=[]
        currentMax=0
        for currentFormula in f:
            buffer=[]
            for x in currentFormula:
                currentMax=max(currentMax,abs(x))
                
                #print("currentMax: ",currentMax)
                if x>0:
                    buffer.append(x+shift)
                else:
                    buffer.append(x-shift)

                #currentRes.append(buffer)

            currentRes.append(buffer) #nik
            #print("buffer: ", buffer)
        
            #recFormulas,recMappings=concat3SATMany(formulas[1:],shift+currentMax,n)
        #finalRes.insert(0,currentRes)  #nik
        finalRes.append(currentRes)
        #finalMapping.append([list(range(shift,shift+n))])
        finalMapping.append(list(range(shift,shift+n))) #nik
        #finalMapping.insert(0,[list(range(shift,shift+n))]) #nik
        shift=shift+currentMax # nik

            #return [currentRes]+recFormulas,[list(range(shift,shift+n))]+recMappings
        #return finalRes, finalMapping
    return finalRes, finalMapping

def concat3SATManyIterativeWrapper(formulas,n):
    bufferFormulas,bufferMappings=concat3SATManyIterative(formulas,0,n)
    res=[]
    for b in bufferFormulas:
        for x in b:
            res.append(x)
    return res,bufferMappings



def add2SATQUBO(m,x0,x1):
    
    q00=[-2, 1, 1, -1, 1, 1]
    q01=[-2, 1, 1, 0, 0, -2]
    q10=[-2, 1, 1, 0, -2, 0]
    q11=[-1, -2, 1, 0, 0, -2]
    
    y0=abs(x0)-1
    y1=abs(x1)-1

    qX=0
    if x0>0 and x1>0:
        qX=q00
    elif x0>0 and x1<0:
        qX=q01
    elif x0<0 and x1>0:
        qX=q10
    elif x0<0 and x1<0:
        qX=q11

    a0=qX[0]
    a1=qX[1]
    a2=qX[2]
    b0=qX[3]
    b1=qX[4]
    b2=qX[5]

    m[y0,y0]=m[y0,y0]+a1*b1+a0*b1+a1*b0
    m[y0,y1]=m[y0,y1]+a1*b2+a2*b1
    m[y1,y1]=m[y1,y1]+a2*b2+a0*b2+a2*b0
    
    return


# These are the relevant factors. The polynomial is 
#
# Add the equation (x0 or x1)=s to the qubo.
# Variables start from 1 and -x is negated variable. s must be positive.
#
def addEquationQUBO(m,x0,x1,s):
    
    p00=[0, 1, 1, -2, 0, 1, 1, -1]
    p01=[1, 1, -1, -2, 1, 1, -1, -1]
    p10=[1, -1, 1, -2, 1, -1, 1, -1]
    p11=[-3, 1, 1, 2, -2, 1, 1, 2]
        
    if s<0:
        print("s must be positive!")
        return
    # Get indices starting from 0
    y0=abs(x0)-1
    y1=abs(x1)-1
    y2=abs(s)-1

    pX=0
    if x0>0 and x1>0:
        pX=p00
    elif x0>0 and x1<0:
        pX=p01
    elif x0<0 and x1>0:
        pX=p10
    elif x0<0 and x1<0:
        pX=p11

    a0=pX[0]
    a1=pX[1]
    a2=pX[2]
    a3=pX[3]
    b0=pX[4]
    b1=pX[5]
    b2=pX[6]
    b3=pX[7]
    # Now add the elements to the matrix.
    m[y0,y0]=m[y0,y0]+a1*b1+a0*b1+a1*b0
    m[y0,y1]=m[y0,y1]+a1*b2+a2*b1
    m[y0,y2]=m[y0,y2]+a1*b3+a3*b1
    m[y1,y1]=m[y1,y1]+a2*b2+a0*b2+a2*b0
    m[y1,y2]=m[y1,y2]+a2*b3+a3*b2
    m[y2,y2]=m[y2,y2]+a3*b3+a0*b3+a3*b0

    return



def convertClause(clause, start):
    if len(clause)<3:
        return [clause],start
    res,maxStart=convertClause([start]+clause[2:],start+1)
    return [[clause[0],clause[1],start]]+res,maxStart


def convertAllClauses(clauses,start):
    currentStart=start
    res=[]
    for c in clauses:
        buffer,currentStart=convertClause(c,currentStart)
        res=res+buffer
    return res,currentStart

def createQUBO(formulas):
    maxVariable=0
    for f in formulas:
        for x in f:
            maxVariable=max(maxVariable,abs(x))
    #print(maxVariable)
    convertedFormulas,maxIndex=convertAllClauses(formulas,maxVariable+1)
    #print(convertedFormulas,maxIndex)
    m=np.zeros((maxIndex-1,maxIndex-1))
    for c in convertedFormulas:
        if len(c)==1: # Replace clause x with x or x
            add2SATQUBO(m,c[0],c[0])
        elif len(c)==2:
            add2SATQUBO(m,c[0],c[1])
        elif len(c)==3:
            addEquationQUBO(m,c[0],c[1],c[2])

    return m


#
# Add a 2SAT formula to the QUBO
# sort y0 and y1 before adding to m
# higher penalty
#


def add2SATQUBOSortPenalty(m,x0,x1,N):
    
    q00=[-2, 1, 1, -1, 1, 1]
    q01=[-2, 1, 1, 0, 0, -2]
    q10=[-2, 1, 1, 0, -2, 0]
    q11=[-1, -2, 1, 0, 0, -2]

    qq00=[N*i for i in q00]
    qq01=[N*i for i in q01]
    qq10=[N*i for i in q10]
    qq11=[N*i for i in q11]
    
    y0=abs(x0)-1
    #print(y0)
    y1=abs(x1)-1
    #print(y1)

    qX=0
    if x0>0 and x1>0:
        qX=qq00
    elif x0>0 and x1<0:
        qX=qq01
    elif x0<0 and x1>0:
        qX=qq10
    elif x0<0 and x1<0:
        qX=qq11

    a0=qX[0]
    a1=qX[1]
    a2=qX[2]
    b0=qX[3]
    b1=qX[4]
    b2=qX[5]

    m[y0,y0]=m[y0,y0]+a1*b1+a0*b1+a1*b0
    if y0<y1:
        m[y0,y1]=m[y0,y1]+a1*b2+a2*b1
    else:
        m[y1,y0]=m[y1,y0]+a1*b2+a2*b1
    m[y1,y1]=m[y1,y1]+a2*b2+a0*b2+a2*b0
    
    return m
    
    
# # sort y0 and y1 before adding to m
# higher penalty

# These are the relevant factors. The polynomial is:


#
# Add the equation (x0 or x1)=s to the qubo.
# Variables start from 1 and -x is negated variable. s must be positive.
#
def addEquationQUBOSortPenalty(m,x0,x1,s,N):
    
    p00=[0, 1, 1, -2, 0, 1, 1, -1]
    p01=[1, 1, -1, -2, 1, 1, -1, -1]
    p10=[1, -1, 1, -2, 1, -1, 1, -1]
    p11=[-3, 1, 1, 2, -2, 1, 1, 2]

    pp00=[N*i for i in p00]
    pp01=[N*i for i in p01]
    pp10=[N*i for i in p10]
    pp11=[N*i for i in p11]
    
    if s<0:
        print("s must be positive!")
        return
    # Get indices starting from 0
    y0=abs(x0)-1
    y1=abs(x1)-1
    y2=abs(s)-1

    pX=0
    if x0>0 and x1>0:
        pX=pp00
    elif x0>0 and x1<0:
        pX=pp01
    elif x0<0 and x1>0:
        pX=pp10
    elif x0<0 and x1<0:
        pX=pp11

    a0=pX[0]
    a1=pX[1]
    a2=pX[2]
    a3=pX[3]
    b0=pX[4]
    b1=pX[5]
    b2=pX[6]
    b3=pX[7]
    # Now add the elements to the matrix.
    m[y0,y0]=m[y0,y0]+a1*b1+a0*b1+a1*b0
    if y0<y1:
        m[y0,y1]=m[y0,y1]+a1*b2+a2*b1
    else:
        m[y1,y0]=m[y1,y0]+a1*b2+a2*b1
    if y0<y2:
        m[y0,y2]=m[y0,y2]+a1*b3+a3*b1
    else:
        m[y2,y0]=m[y2,y0]+a1*b3+a3*b1
    m[y1,y1]=m[y1,y1]+a2*b2+a0*b2+a2*b0
    if y1<y2:
        m[y1,y2]=m[y1,y2]+a2*b3+a3*b2
    else:
        m[y2,y1]=m[y2,y1]+a2*b3+a3*b2
    m[y2,y2]=m[y2,y2]+a3*b3+a0*b3+a3*b0
    
    return m


# dwave classical gradient descent solver

def dwaveGreedySolver(m,N):
    
    nonzeros=0
    mDict={}
    for i in range(len(m)):
        for j in range(len(m)):
            if abs(m[i,j])>0.01:
                nonzeros=nonzeros+1
                mDict[i,j]=m[i,j]
    print("#nonzeros/#quboEntries",nonzeros,"/",len(m)**2)
    print("sparsity",nonzeros/len(m)**2)
    
    mDict2={}
    mDict2Pair={}
    for x in mDict:
        if x[0]==x[1]:
            mDict2['x'+str(x[0])]=mDict[x]
        else:
            mDict2Pair['x'+str(x[0]),'x'+str(x[1])]=mDict[x]
    bqm=dimod.BinaryQuadraticModel(mDict2,mDict2Pair,"BINARY")
    
    solver=greedy.SteepestDescentSolver()
    sampleset=solver.sample(bqm,num_reads=N)
    best=sampleset.first
    res=[]
    for i in range(len(best.sample)):
        res.append(best.sample['x'+str(i)])
    #print(best.energy,[res[0],res[2],res[4],res[6],res[8]],[res[47],res[49],res[51],res[53],res[55]])
    #print(best.energy,res)

    
    return best.energy,res



# ## Qubo including all relevant parameter

# create qubo with costs, demand constraint, minup constraint, startup costs, min down

def createQUBOCostMaxgenDemandMinupMindownMaxupMaxDownStartcostIterative(cost, maxgen, demand, minup, mindown, maxup, maxdown, startcost):

    # input check
    if len(cost)!=len(startcost):
        print("wrong cost/startup input parameter")
        return
    
    if len(maxgen)!=len(cost):
        print("wrong cost/supply input parameter")
        return
    
    
    numberOfTimeSteps=len(demand)
    numberOfUnits=len(cost)

# setting up sat formulas and mapping ITERATIVE functions

    sat=[]
    for i in range(len(cost)):
        sat.append(generateSAT(mindown[i], maxdown[i], minup[i], maxup[i],numberOfTimeSteps))

    formulas, maps=concat3SATManyIterativeWrapper(sat,numberOfTimeSteps)
    print("Number of formulas: ",len(formulas))

# constraint qubo
    
    maxVariable=0
    for f in formulas:
        for x in f:
            maxVariable=max(maxVariable,abs(x))
#    print(maxVariable)
    convertedFormulas,maxIndex=convertAllClauses(formulas,maxVariable+1)
#    print(convertedFormulas,maxIndex)
    m=np.zeros((maxIndex-1,maxIndex-1))
    for c in convertedFormulas:
        if len(c)==1: # Replace clause x with x or x
            add2SATQUBO(m,c[0],c[0])
        elif len(c)==2:
            add2SATQUBO(m,c[0],c[1])
        elif len(c)==3:
            addEquationQUBO(m,c[0],c[1],c[2])
#------------
# Adding Costs

    for i in range(numberOfUnits):
        currentMapping=maps[i]
        currentCost=cost[i]
        currentstartupCost=startcost[i]
        
        for j in currentMapping:
            m[j,j]=m[j,j]+currentCost*maxgen[i]    # costs
            
        for j in currentMapping[1:]:
            m[j,j]=m[j,j]+currentstartupCost       # startcost
            m[j-1,j]=m[j-1,j]-currentstartupCost   # startcost

# Create the conditions for demand satisfaction (each power station can create maxgen[i] units)
# See formula above for (a_i+b_i-d_i)^2

    penaltyDemand=100

    for t in range(numberOfTimeSteps):
        mappingsThisTimeStep=[]
        for k in maps:
            mappingsThisTimeStep.append(k[t])

        for ix in range(len(mappingsThisTimeStep)):
            x=mappingsThisTimeStep[ix]
#            m[x,x]=m[x,x]+penaltyDemand*(1-2*demand[t])
            m[x,x]=m[x,x]+penaltyDemand*(maxgen[ix]**2-2*demand[t]*maxgen[ix])
            for iy in range(ix+1,len(mappingsThisTimeStep)):
                y=mappingsThisTimeStep[iy]
#                m[x,y]=m[x,y]+penaltyDemand*2
                m[x,y]=m[x,y]+2*penaltyDemand*maxgen[ix]*maxgen[iy]


    return m, maps


# generate qubo (dict) from numpy array
def matrix2qubo(m):

    qubo={}
    for i in range(len(m)):
        for j in range(len(m)):
            if abs(m[i,j])>0.0:
                qubo[(i,j)]=m[i,j]

    return qubo



# ADAPTIVE PENALTY TERMS PDemand,PCost,PStart,PConstr
## createSATqubo(cost,maxgen,demand,minup,mindown,maxup,maxdown,startcost,PDemand,PCost,PStart,PConstr)
def createSATquboPenalty(cost,maxgen,demand,minup,mindown,maxup,maxdown,startcost,PDemand=100,PCost=10,PStart=1,PConstr=1,verbose=True):

    # input check
    if len(cost)!=len(startcost):
        print("wrong cost/startup input parameter")
        return
    
    if len(maxgen)!=len(cost):
        print("wrong cost/supply input parameter")
        return
    
    
    numberOfTimeSteps=len(demand)
    numberOfUnits=len(cost)

# setting up sat formulas and mapping ITERATIVE functions

    sat=[]
    for i in range(len(cost)):
        sat.append(generateSAT(mindown[i], maxdown[i], minup[i], maxup[i],numberOfTimeSteps))

    formulas, maps=concat3SATManyIterativeWrapper(sat,numberOfTimeSteps)
    if verbose:
        print("Number of formulas: ",len(formulas))

# constraint qubo
    
    maxVariable=0
    for f in formulas:
        for x in f:
            maxVariable=max(maxVariable,abs(x))
#    print(maxVariable)
    convertedFormulas,maxIndex=convertAllClauses(formulas,maxVariable+1)
#    print(convertedFormulas,maxIndex)
    m=np.zeros((maxIndex-1,maxIndex-1))
    for c in convertedFormulas:
        if len(c)==1: # Replace clause x with x or x
            add2SATQUBOSortPenalty(m,c[0],c[0],PConstr)
        elif len(c)==2:
            add2SATQUBOSortPenalty(m,c[0],c[1],PConstr)
        elif len(c)==3:
            addEquationQUBOSortPenalty(m,c[0],c[1],c[2],PConstr)
#------------
# Adding Costs

    penaltyCost=PCost
    penaltyStart=PStart

    for i in range(numberOfUnits):
        currentMapping=maps[i]
        currentCost=cost[i]
        currentstartupCost=startcost[i]
        
        for j in currentMapping:
            m[j,j]=m[j,j]+currentCost*maxgen[i]*penaltyCost    # costs
            
        for j in currentMapping[1:]:
            m[j,j]=m[j,j]+currentstartupCost*penaltyStart       # startcost
            m[j-1,j]=m[j-1,j]-currentstartupCost*penaltyStart   # startcost

# Create the conditions for demand satisfaction (each power station can create maxgen[i] units)
# See formula above for (a_i+b_i-d_i)^2

    penaltyDemand=PDemand

    for t in range(numberOfTimeSteps):
        mappingsThisTimeStep=[]
        for k in maps:
            mappingsThisTimeStep.append(k[t])

        for ix in range(len(mappingsThisTimeStep)):
            x=mappingsThisTimeStep[ix]
#            m[x,x]=m[x,x]+penaltyDemand*(1-2*demand[t])
            m[x,x]=m[x,x]+penaltyDemand*(maxgen[ix]**2-2*demand[t]*maxgen[ix])
            for iy in range(ix+1,len(mappingsThisTimeStep)):
                y=mappingsThisTimeStep[iy]
#                m[x,y]=m[x,y]+penaltyDemand*2
                m[x,y]=m[x,y]+2*penaltyDemand*maxgen[ix]*maxgen[iy]


    return m, maps


#
# Add a 2SAT formula to the QUBO
# sort y0 and y1 before adding to m
#


def add2SATQUBOSort(m,x0,x1):

    q00=[-2, 1, 1, -1, 1, 1]
    q01=[-2, 1, 1, 0, 0, -2]
    q10=[-2, 1, 1, 0, -2, 0]
    q11=[-1, -2, 1, 0, 0, -2]


    y0=abs(x0)-1
    print(y0)
    y1=abs(x1)-1
    print(y1)

    qX=0
    if x0>0 and x1>0:
        qX=q00
    elif x0>0 and x1<0:
        qX=q01
    elif x0<0 and x1>0:
        qX=q10
    elif x0<0 and x1<0:
        qX=q11

    a0=qX[0]
    a1=qX[1]
    a2=qX[2]
    b0=qX[3]
    b1=qX[4]
    b2=qX[5]

    m[y0,y0]=m[y0,y0]+a1*b1+a0*b1+a1*b0
    if y0<y1:
        m[y0,y1]=m[y0,y1]+a1*b2+a2*b1
    else:
        m[y1,y0]=m[y1,y0]+a1*b2+a2*b1
    m[y1,y1]=m[y1,y1]+a2*b2+a0*b2+a2*b0



#
# Add the equation (x0 or x1)=s to the qubo.
# Variables start from 1 and -x is negated variable. s must be positive.
#
def addEquationQUBOSort(m,x0,x1,s):

    # # sort y0 and y1 before adding to m

    # These are the relevant factors. The polynomial is:
    p00=[0, 1, 1, -2, 0, 1, 1, -1]
    p01=[1, 1, -1, -2, 1, 1, -1, -1]
    p10=[1, -1, 1, -2, 1, -1, 1, -1]
    p11=[-3, 1, 1, 2, -2, 1, 1, 2]


    if s<0:
        print("s must be positive!")
        return
    # Get indices starting from 0
    y0=abs(x0)-1
    y1=abs(x1)-1
    y2=abs(s)-1

    pX=0
    if x0>0 and x1>0:
        pX=p00
    elif x0>0 and x1<0:
        pX=p01
    elif x0<0 and x1>0:
        pX=p10
    elif x0<0 and x1<0:
        pX=p11

    a0=pX[0]
    a1=pX[1]
    a2=pX[2]
    a3=pX[3]
    b0=pX[4]
    b1=pX[5]
    b2=pX[6]
    b3=pX[7]
    # Now add the elements to the matrix.
    m[y0,y0]=m[y0,y0]+a1*b1+a0*b1+a1*b0
    if y0<y1:
        m[y0,y1]=m[y0,y1]+a1*b2+a2*b1
    else:
        m[y1,y0]=m[y1,y0]+a1*b2+a2*b1
    if y0<y2:
        m[y0,y2]=m[y0,y2]+a1*b3+a3*b1
    else:
        m[y2,y0]=m[y2,y0]+a1*b3+a3*b1
    m[y1,y1]=m[y1,y1]+a2*b2+a0*b2+a2*b0
    if y1<y2:
        m[y1,y2]=m[y1,y2]+a2*b3+a3*b2
    else:
        m[y2,y1]=m[y2,y1]+a2*b3+a3*b2
    m[y2,y2]=m[y2,y2]+a3*b3+a0*b3+a3*b0


# using sorted 2SAT, SATFormulas
# create qubo with costs, demand constraint, minup constraint, startup costs, min down

def createSATqubo(cost, maxgen, demand, minup, mindown, maxup, maxdown, startcost):

    # input check
    if len(cost)!=len(startcost):
        print("wrong cost/startup input parameter")
        return
    
    if len(maxgen)!=len(cost):
        print("wrong cost/supply input parameter")
        return
    
    
    numberOfTimeSteps=len(demand)
    numberOfUnits=len(cost)

# setting up sat formulas and mapping ITERATIVE functions

    sat=[]
    for i in range(len(cost)):
        sat.append(generateSAT(mindown[i], maxdown[i], minup[i], maxup[i],numberOfTimeSteps))

    formulas, maps=concat3SATManyIterativeWrapper(sat,numberOfTimeSteps)
    print(formulas)
    print("Number of formulas: ",len(formulas))

# constraint qubo
    
    maxVariable=0
    for f in formulas:
        for x in f:
            maxVariable=max(maxVariable,abs(x))
            
    print("maxVariable: ",maxVariable)
    convertedFormulas,maxIndex=convertAllClauses(formulas,maxVariable+1)
    print("convertedFormulas: ",convertedFormulas)
    #print("convertedFormulas ",convertedFormulas," maxIndex ",maxIndex)
    m=np.zeros((maxIndex-1,maxIndex-1))
    for c in convertedFormulas:
        #print(c)
        if len(c)==1: # Replace clause x with x or x
            add2SATQUBOSort(m,c[0],c[0])
            #print("add2SATQUBOSort: ",add2SATQUBOSort(m,c[0],c[0]))
        elif len(c)==2:
            add2SATQUBOSort(m,c[0],c[1])
            #print("add2SATQUBOSort: ",add2SATQUBOSort(m,c[0],c[1]))
        elif len(c)==3:
            addEquationQUBOSort(m,c[0],c[1],c[2])
            #print("addEquationQUBOSort: ",addEquationQUBOSort(m,c[0],c[1],c[2]))
            
    print("Constraint QUBO:\n ",m)
#------------
# Adding Costs

    for i in range(numberOfUnits):
        currentMapping=maps[i]
        currentCost=cost[i]
        currentstartupCost=startcost[i]
        
        for j in currentMapping:
            m[j,j]=m[j,j]+currentCost*maxgen[i]    # costs
        #print("Costs: ",m[j,j])
            
        for j in currentMapping[1:]:
            m[j,j]=m[j,j]+currentstartupCost       # startcost
            #print(m[j,j])
            m[j-1,j]=m[j-1,j]-currentstartupCost   # startcost
            #print(m[j-1,j])
        #print("Startcosts: ",m[j,j])
        
    print("Cost QUBO:\n ",m)

# Create the conditions for demand satisfaction (each power station can create maxgen[i] units)
# See formula above for (a_i+b_i-d_i)^2

    penaltyDemand=100
    print("penaltyDemand: ",penaltyDemand)

    for t in range(numberOfTimeSteps):
        mappingsThisTimeStep=[]
        for k in maps:
            mappingsThisTimeStep.append(k[t])

        for ix in range(len(mappingsThisTimeStep)):
            x=mappingsThisTimeStep[ix]
#            m[x,x]=m[x,x]+penaltyDemand*(1-2*demand[t])
            m[x,x]=m[x,x]+penaltyDemand*(maxgen[ix]**2-2*demand[t]*maxgen[ix])
        #print("Demand: ",m[x,x])
            for iy in range(ix+1,len(mappingsThisTimeStep)):
                y=mappingsThisTimeStep[iy]
#                m[x,y]=m[x,y]+penaltyDemand*2
                m[x,y]=m[x,y]+2*penaltyDemand*maxgen[ix]*maxgen[iy]
                #print("Demand: ",m[x,y])
    print("Demand QUBO:\n  ",m)

    return m, maps


## verify solutions by checking the solution vector for minup, mindown, maxup, maxdown constraints

### Calculate minimum sequence length
def tokenizerBy0(str, bufferSoFar, resSoFar):
    if len(str)==0:
        if len(bufferSoFar)>0:
            resSoFar.append(bufferSoFar)
        return resSoFar
    if str[0]=='0':
        if len(bufferSoFar)>0:
            resSoFar.append(bufferSoFar)
        return tokenizerBy0(str[1:],[],resSoFar)
    else:
        bufferSoFar.append(str[0])
        return tokenizerBy0(str[1:],bufferSoFar,resSoFar)

### Calculate maximum sequence length
def tokenizerBy1(str, bufferSoFar, resSoFar):
    if len(str)==0:
        if len(bufferSoFar)>0:
            resSoFar.append(bufferSoFar)
        return resSoFar
    if str[0]=='1':
        if len(bufferSoFar)>0:
            resSoFar.append(bufferSoFar)
        return tokenizerBy1(str[1:],[],resSoFar)
    else:
        bufferSoFar.append(str[0])
        return tokenizerBy1(str[1:],bufferSoFar,resSoFar)

#
# minup: shortest 1 sequence
#
def minupSequence0(str):
    buffer=tokenizerBy0('0'+str+'0',[],[])
    if len(buffer)==0:
        return 0
    else:
        #buffer2=[len(x) for x in buffer]
        #print(buffer2)
        return min([len(x) for x in buffer])

#
# mindown: shortest 0 sequence
#
def mindownSequence0(str):
    buffer=tokenizerBy1('1'+str+'1',[],[])
    return min([len(x) for x in buffer])

#
# maxup: longest 1 sequence
#
def maxupSequence0(str):
    buffer=tokenizerBy0('0'+str+'0',[],[])
    return max([len(x) for x in buffer])

#
# maxdown: longest 0 sequence
#
def maxdownSequence0(str):
    buffer=tokenizerBy1('1'+str+'1',[],[])
    return max([len(x) for x in buffer])



def checkMinupMinDownMaxupMaxdownChecks(minup,mindown,maxup,maxdown,bufferSupply,TotalSupply,demand):
    str2=[]
    for j in range(len(bufferSupply)):
        str2.append(''.join(str(i) for i in bufferSupply[j]))

    print(str2)

    # Calculate NumberOfConstraintViolations
    minup_violate=0
    mindown_violate=0
    maxup_violate=0
    maxdown_violate=0
    total_violate=0
    num_constraints=0
    ratio_violate=0

#    print("--------------------------")
#    print("Check Minup constraint: ")
    # check Minup
    for i in range(len(str2)):
        if (minup[i] == 0):
            print("no minup constraint")
        else:
            if (minupSequence0(str2[i]) == 0):
                print("unit: ",i," --> Unit Off")
            elif (minupSequence0(str2[i]) >= minup[i]):
                print("unit: ",i," --> Minup Correct")
            else:
                print("unit: ",i," --> Minup VIOLATION",minupSequence0(str2[i])-minup[i])
                minup_violate=minup_violate+1
#    print("Number of minup violations:   ",minup_violate)

#    print("--------------------------")
#    print("Check Mindown constraint: ")
    # check Mindown
    for i in range(len(str2)):
        if (mindown[i] == 0):
            print("no mindown constraint")
        else:
            if (mindownSequence0(str2[i]) == mindown[i]):
                print("unit: ",i," --> Mindown Correct")
            elif (mindownSequence0(str2[i]) == 0):
                print("unit: ",i," --> Unit Off")
            else:
                print("unit: ",i," --> Mindown VIOLATION",mindownSequence0(str2[i])-mindown[i])
                mindown_violate=mindown_violate+1
#    print("Number of mindown violations: ",mindown_violate)

#    print("--------------------------")
#    print("Check Maxup constraint: ")
    # check Maxup
    for i in range(len(str2)):
        if (maxup[i] == 0):
            print("no maxup constraint")
        else:
            if (maxupSequence0(str2[i]) == maxup[i]):
                print("unit: ",i," --> Maxup Correct")
            else:
                print("unit: ",i," --> Maxup VIOLATION",maxupSequence0(str2[i])-maxup[i])
                maxup_violate=maxup_violate+1
#    print("Number of maxup violations:   ",maxup_violate)

#    print("--------------------------")
#    print("Check Maxdown constraint: ")
    # check Maxdown
    for i in range(len(str2)):
        if (maxdown[i] == 0):
            print("no maxdown constraint")
        else:
            if (maxdownSequence0(str2[i]) == maxdown[i]):
                print("unit: ",i," --> Maxdown Correct")
            else:
                print("unit: ",i," --> Maxdown VIOLATION",maxdownSequence0(str2[i])-maxdown[i])
                maxdown_violate=maxdown_violate+1
#    print("Number of maxdown violations: ",maxdown_violate)

    ## CHECK DEMAND CONSTRAINT
    demand_constraint=0
    unmatched_demand=[]
    unmatched_demand_ratio=[]

    for i in range(len(demand)):

        if len(demand) != len(TotalSupply):
            print("length demand and supply do not match")

        elif demand[i]>TotalSupply[i]:
            #print("DEMAND ummatched --> demand constrain break")
            print("unmatched demand       = ",demand[i]-TotalSupply[i])
            print("unmatched demand ratio = ",TotalSupply[i]/demand[i])
            demand_constraint=demand_constraint+1
            unmatched_demand.append(demand[i]-TotalSupply[i])
            unmatched_demand_ratio.append(TotalSupply[i]/demand[i])

        elif demand[i]<TotalSupply[i]:
            #print("Over supply      --> demand constrain break")
            print("over supply            = ",TotalSupply[i]-demand[i])
            print("unmatched demand ratio = ",TotalSupply[i]/demand[i])
            demand_constraint=demand_constraint+1
            unmatched_demand.append(demand[i]-TotalSupply[i])
            unmatched_demand_ratio.append(TotalSupply[i]/demand[i])

        elif demand[i]==TotalSupply[i]:
            print("demand                 = supply  --> constraint matched")
            print("unmatched demand ratio = ",TotalSupply[i]/demand[i])
            print("Demand                 = ",demand[i],"| Supply = ",TotalSupply[i])
            unmatched_demand.append(0)
            unmatched_demand_ratio.append(0)

    print("Number of demand constraint breaks = ",demand_constraint)
    
    total_violate=(minup_violate+mindown_violate+maxup_violate+maxdown_violate+demand_constraint)
#    print("Number of TOTAL violations:   ",total_violate)
    
    num_constraints=((len(minup)+len(mindown)+len(maxup)+len(maxdown))*len(demand))+demand_constraint
    ratio_violate="{:.1%}".format(1-total_violate/num_constraints)
#    print("Quality (violation/total constraints): ",ratio_violate)
    
    return total_violate, ratio_violate, num_constraints, unmatched_demand, unmatched_demand_ratio




# print Solution in a data frame for re-usability

def printSolution(res, maps, cost, minup, demand, maxgen):

    n=len(demand)
    solutionVector=[]
    solutionCostTotal=0
    solutionCost=[]
    solutionSupply=[]
    costsPerTimeStep=[]
    
    for m in maps:
        time=[]
        buffer1=[]

        ti=0
        for t in m:
            ti=ti+1
            time.append(ti)
        
            if res[t]==1:
                buffer1.append(1)

            if res[t]==0:
                buffer1.append(0)

        solutionVector.append(buffer1)
    #print(n," time steps"," | solutionVector: ",solutionVector)

    for i in range(len(solutionVector)):
        solutionCostTotal=solutionCostTotal+cost[i]*sum(solutionVector[i])
#        solutionCost=solutionCost+cost[i]*solutionVector[i]
#        print("Costs of production: ", solutionCostTotal)
    
    print("Total costs of production: ", solutionCostTotal)

# OLD Version to calculate Supply and Cost per Unit
#    for i in range(len(solutionVector)):
#        for j in range(len(solutionVector[i])):
#            solutionCost.append(cost[i]*solutionVector[i][j]) # calculate costs
#            solutionSupply.append(maxgen[i]*solutionVector[i][j]) # calculate maxgen
#    print("Costs of production: ", solutionCost)
#    print("Supply: ", solutionSupply)

# costs per unit and time step
    bufferCost=[]
#    for i in range(len(cost)):
#        bufferCost.append(np.dot(solutionVector[i],cost[i]))
    for i in range(len(solutionVector)):
         bufferCost.append([cost[i] * x for x in solutionVector[i]])

# supply per unit and time step
    bufferSupply=[]
#    for i in range(len(maxgen)):
#        bufferSupply.append(np.dot(solutionVector[i],maxgen[i]))
    
    for i in range(len(solutionVector)):
        bufferSupply.append([maxgen[i] * x for x in solutionVector[i]])

    #print("Cost   = ",bufferCost)
    #print("Supply = ",bufferSupply)
#    for i in time:

#        for j in cost:
#            solutionCost[i]=cost[j]*solutionVector[j]
#            print("Cost of unit: ",j,":", solutionCost[j])

    #Solution dataFrame
    B1 = np.reshape(solutionVector, (len(cost),len(demand)))
    ps1=['Station '+ str(j) for j in range(len(solutionVector))]
    st1=['Step '+ str(j) for j in time]
    df1=pd.DataFrame(data=B1.transpose(),index=st1,columns=ps1)
#    costsPerTimeStep = ['Cost '+ str(j) for j in range(len(solutionVector))]
#    df['Cost'] = costsPerTimeStep
    df1['demand'] = demand
    TotalSupply=[sum(x) for x in zip(*bufferSupply)]
    df1['TotalSupply'] = TotalSupply
#    df['supply'] = bufferSupply

    #Supply dataFrame
    B2 = np.reshape(bufferSupply, (len(cost),len(demand)))
    ps2=['Station supply '+ str(j) for j in range(len(bufferSupply))]
    #data2 = np.array(bufferSupply)
    df2 = pd.DataFrame(data=B2.transpose(), columns=ps2)
    #df['supply'] = bufferSupply
    #print(df2)

    #Total supply per time step dataFrame
    #B3 = np.reshape(bufferSupply, (len(cost),len(demand)))
    #ps3=['Total supply '+ str(j) for j in range(len(bufferSupply))]
    #data2 = np.array(bufferSupply)
    #df3 = pd.DataFrame(data=B3.transpose(), columns=ps3)

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    #df3.reset_index(drop=True, inplace=True)

    #dffull = pd.concat( [df1, df2,df3], axis=1)
    dffull = pd.concat( [df1, df2], axis=1)

    #print(df)
    return  dffull, solutionVector, bufferSupply, solutionCostTotal, TotalSupply


# print PYGRND SOLVER (JoS QUANTUM's quantum inspired solver) Solution in a data frame for re-usability
def printPYGRNDSolution(qubo_solution, maps, cost, minup, demand, maxgen):
    n=len(demand)
    solutionVector=[]
    solutionCostTotal=0
    solutionCost=[]
    solutionSupply=[]
    costsPerTimeStep=[]
    
    for m in maps:
        time=[]
        buffer1=[]

        ti=0
        for t in m:
            ti=ti+1
            time.append(ti)
        
            if qubo_solution[t]==1:
                buffer1.append(1)

            if qubo_solution[t]==0:
                buffer1.append(0)

        solutionVector.append(buffer1)
#    print(n," time steps"," | solutionVector: ",solutionVector)

    for i in range(len(solutionVector)):
        solutionCostTotal=solutionCostTotal+cost[i]*sum(solutionVector[i])
#        solutionCost=solutionCost+cost[i]*solutionVector[i]
#        print("Costs of production: ", solutionCostTotal)
    
#    print("Total costs of production: ", solutionCostTotal)

# OLD Version to calculate Supply and Cost per Unit
#    for i in range(len(solutionVector)):
#        for j in range(len(solutionVector[i])):
#            solutionCost.append(cost[i]*solutionVector[i][j]) # calculate costs
#            solutionSupply.append(maxgen[i]*solutionVector[i][j]) # calculate maxgen
#    print("Costs of production: ", solutionCost)
#    print("Supply: ", solutionSupply)

# costs per unit and time step
    bufferCost=[]
#    for i in range(len(cost)):
#        bufferCost.append(np.dot(solutionVector[i],cost[i]))
    for i in range(len(solutionVector)):
         bufferCost.append([ cost[i] * x for x in solutionVector[i]])

# supply per unit and time step
    bufferSupply=[]
#    for i in range(len(maxgen)):
#        bufferSupply.append(np.dot(solutionVector[i],maxgen[i]))
    
    for i in range(len(solutionVector)):
        bufferSupply.append([ maxgen[i] * x for x in solutionVector[i]])

#    print("Cost   = ",bufferCost)
#    print("Supply = ",bufferSupply)
#    for i in time:

#        for j in cost:
#            solutionCost[i]=cost[j]*solutionVector[j]
#            print("Cost of unit: ",j,":", solutionCost[j])

    #Solution dataFrame
    B1 = np.reshape(solutionVector, (len(cost),len(demand)))
    ps1=['Station '+ str(j) for j in range(len(solutionVector))]
    st1=['Step '+ str(j) for j in time]
    df1=pd.DataFrame(data=B1.transpose(),index=st1,columns=ps1)
#    costsPerTimeStep = ['Cost '+ str(j) for j in range(len(solutionVector))]
#    df['Cost'] = costsPerTimeStep
    df1['demand'] = demand
    TotalSupply=[sum(x) for x in zip(*bufferSupply)]
    df1['TotalSupply'] = TotalSupply
#    df['supply'] = bufferSupply

    #Supply dataFrame
    B2 = np.reshape(bufferSupply, (len(cost),len(demand)))
    ps2=['Station supply '+ str(j) for j in range(len(bufferSupply))]
#    data2 = np.array(bufferSupply)
    df2 = pd.DataFrame(data=B2.transpose(), columns=ps2)
    #df['supply'] = bufferSupply
    #print(df2)

    #Total supply per time step dataFrame
    #B3 = np.reshape(bufferSupply, (len(cost),len(demand)))
    #ps3=['Total supply '+ str(j) for j in range(len(bufferSupply))]
    #data2 = np.array(bufferSupply)
    #df3 = pd.DataFrame(data=B3.transpose(), columns=ps3)

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    #df3.reset_index(drop=True, inplace=True)

    #dffull = pd.concat( [df1, df2,df3], axis=1)
    dffull = pd.concat( [df1, df2], axis=1)

    #print(df)
    return  dffull, solutionVector, bufferSupply, solutionCostTotal, TotalSupply
