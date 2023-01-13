import numpy as np
import math
import random
import itertools
import scipy
from scipy.linalg import block_diag
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import dimod
import neal
import greedy

from azure.quantum.optimization import Problem, ProblemType, Term
from azure.quantum import Workspace


def penaltycheck(p,dgen,mingen,start,minup,mindown,on,T,n,pres):
    
    """
    function penaltycheck checks if constraints are violated by checking the binary solution-vector "p"
    """
    
    pmincheck=True
    oncheck=True
    startcheck=True
    minupcheck=True
    mindowncheck=True
    okay=True
    minupviolation=0
    mingenviolation=0
    power=[[sum(np.fromiter((p[t*n*(pres+2)+i*pres+r]*dgen[i][r] for r in range(pres)), float)) for i in range(n)] for t in range(T)]  #power supply=binary vector + power stages (given by dgen)
    for t in range(T):
        for i in range(n):
            power[t][i]=power[t][i]+p[t*n*(pres+2)+n*pres+i]*mingen[i]    #adding mingen to power supply
    for t in range(T):
        for i in range(n):
            if power[t][i]>0:
                if (power[t][i]+mingen[i])<mingen[i]:            #check mingen
                    pmincheck=False
                    print('pmincheck')
                    mingenviolation=mingenviolation+1
                if on[t][i]==0:                                  #check behavior of "on" variable
                    oncheck=False
            if power[t][i]==0:
                if on[t][i]>0:                                   #check behavior of "on" variable
                    oncheck=False
                if start[t][i]>0:                                #check behavior of "start" variable
                    startcheck=False

            if t==0:
                if power[t][i]>0:                                #check start
                    if start[t][i]==0:
                        startcheck=False
                if power[t][i]>0:                                #check minup
                    if t+minup[i]>T:
                        endt=T
                    else:
                        endt=t+minup[i]
                    for tau in range(t,endt):
                        if power[tau][i]==0:
                            minupcheck=False
            else:
                if power[t][i]>0 and power[t-1][i]==0:            #check start
                    if start[t][i]==0:
                        startcheck=False
                    if t+minup[i]>T:
                        endt=T
                    else:
                        endt=t+minup[i]
                    for tau in range(t,endt):
                        if power[tau][i]==0:
                            minupviolation=minupviolation+1
                            minupcheck=False
                if power[t-1][i]>0 and power[t][i]==0:            #check mindown
                    if t+mindown[i]>T:
                        enddown=T
                    else:
                        enddown=t+mindown[i]
                    for tau in range(t,enddown):
                        if power[tau][i]>0:
                            mindowncheck=False
    print('pmincheck: ',pmincheck)
    print('minupcheck: ',minupcheck)
    print('mindowncheck: ',mindowncheck)
    print('oncheck: ',oncheck)
    print('startcheck: ',startcheck)
    if pmincheck==False or minupcheck==False or mindowncheck==False or oncheck==False or startcheck==False:
        okay=False
    quality=1-(minupviolation+mingenviolation)/(n*T)
    return okay, pmincheck, oncheck, startcheck, minupcheck, mindowncheck, quality


def costsummation(p,dgen,mingen,varcost,startcost,T,n,pres): 

    """
    function to calculate the costs from a binary solution vector "p"  #dgen hold the discretized power stages of the units
    """

    runcosts=0
    startcosts=0
    costs=0
    powersup=[[sum(np.fromiter((p[t*n*(pres+2)+i*pres+r]*dgen[i][r] for r in range(pres)), float)) for i in range(n)] for t in range(T)]
    for t in range(T):
        for i in range(n):
            powersup[t][i]=powersup[t][i]+mingen[i]*p[t*n*(pres+2)+n*pres+i]
    for t in range(T):
        for i in range(n):
            runcosts=runcosts+powersup[t][i]*varcost[i]
            if t<T-1:
                if t==0:
                    if powersup[t][i]>0:
                        startcosts=startcosts+startcost[i]
                    if powersup[t][i]==0 and powersup[t+1][i]>0:
                        startcosts=startcosts+startcost[i]
                else:
                    if powersup[t][i]==0 and powersup[t+1][i]>0:
                        startcosts=startcosts+startcost[i]
    costs=startcosts+runcosts

    return costs



def BruteForceUCPqubo(M,dgen,varcost,startcost,mingen,minup,mindown,T,n,pres,bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, boolcheck,graphicsout):

    """
    Solve QUBO brute force by iterating through every possible 2^N combinations of the solution vector x^T Q x and checking the objective

    """

    starttime = time.process_time()
    m=len(M)                                                                                      #check all possible combinations for solution vector and search global minimum
    nonzeros=0
    for i in range(m):
        for j in range(m):
            if abs(M[i,j])>0.01:
                nonzeros=nonzeros+1
    if graphicsout==True:
        print("#nonzeros/#quboEntries",nonzeros,"/",m**2)
        print("sparsity",1-nonzeros/m**2)
        print('QUBO size: ',str(m)+' x '+str(m))
    minimum_value=0
    minimum_vector= np.squeeze(np.asarray(np.matrix([[0]*m])))
    tuples=itertools.product(*[(0, 1)]*m)
    saveresultsprice=[]
    saveresultsobjective=[]
    saveresults=[]
    errors=[]
    combined=[]
    for t in tqdm(tuples):
        v=np.matrix(list(t))
        vector=np.array(v[0]).flatten()
        p = np.zeros((T, n, pres))
        power=np.zeros((T,n))
        start=np.zeros((T,n))
        on=np.zeros((T,n))
        combi=np.zeros(2)
        for t in range(T):
            for i in range(n):
                for r in range(pres):
                    power[t][i] = power[t][i] + vector[t*n*(pres+2)+i*pres+r]
        for t in range(T):
            for i in range(n):
                on[t][i]=vector[t*n*(pres+2)+n*(pres)+i]
                start[t][i]=vector[t*n*(pres+2)+n*(pres+1)+i]
                if power[t][i]>0:
                    for r in range(pres):
                        p[t][i][r]=vector[t*n*(pres+2)+i*pres+r]
        okay, pmincheck, oncheck, startcheck, minupcheck, minwoncheck, errorquality = penaltycheck(vector,dgen,mingen,start,minup,mindown,on,T,n,pres)
        errors.append(errorquality)
        price=costsummation(vector,dgen,mingen,varcost,startcost,T,n,pres)
        combi=[price,okay]
        saveresultsprice.append(combi)

        res=np.matmul(np.matmul(v,M),np.transpose(v))[0,0]
        combiobjective=[res,okay]
        saveresultsobjective.append(combiobjective)
        if res<minimum_value:
            if okay==boolcheck:
                if r<bestobjective:
                    bestobjective=r
                    bestobjectiveAns=vector
                    bestobjectiveprice=price
                if price<bestprice:
                    bestprice=price
                    bestpriceAns=vector
                    bestpriceobj=r
            minimum_value=res
            minimum_vector=np.array(v[0]).flatten()
        combiobjpriceokay=[res,price,errorquality]
        combined.append(np.array(combiobjpriceokay))
    timetosolve=time.process_time() - starttime                                               # calculate time for problem saving
    if graphicsout==True:
        print('time to brute solve the best solution:',timetosolve)

        fig1, ax1 = plt.subplots()                                                            # plot energy landscape and costs of all possible solutions

        ax2 = ax1.twinx()
        ax1.plot([k for k in range(len(saveresultsobjective))],[saveresultsobjective[k][0] for k in range(len(saveresultsobjective))],'b-')
        ax2.plot([k for k in range(len(saveresultsprice))],[saveresultsprice[k][0] for k in range(len(saveresultsprice))],'rx')

        ax1.set_xlabel('states', fontsize=20)
        ax1.set_ylabel('objective value', color='b', fontsize=20)
        ax2.set_ylabel('cost value', color='r', fontsize=20)
        plt.show()

    return minimum_value, minimum_vector, combined, saveresultsprice, saveresultsobjective, errors, bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, timetosolve




def MonteCarloUCPqubo(Q,N,dgen,varcost,startcost,mingen,minup,mindown,T,n,pres,bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, boolcheck,graphicsout):

    """
    Monte Carlo solver that randomly samples from solution space N times a given number of binary vectors and checks if constraints are fulfilled
    """

    starttime = time.process_time()
    m=len(Q)
    nonzeros=0
    saveresultsprice=[]
    saveresultsobjective=[]
    errors=[]
    for i in range(m):
        for j in range(m):
            if abs(Q[i,j])>0.01:
                nonzeros=nonzeros+1
    if graphicsout==True:
        print("#nonzeros/#quboEntries",nonzeros,"/",m**2)
        print("sparsity",1-nonzeros/m**2)
        print('QUBO size: ',str(m)+' x '+str(m))

    # Find good solutions and map them back.
    cheapestPrice=float('inf')
    vector=0
    combined=[]
    for i in tqdm(range(N)):
        v=np.zeros((1,len(Q)))
        for i in range(len(Q)):
            v[0,i]=round(random.random())
        vector=np.array(v[0]).flatten()
        p = np.zeros((T, n, pres))
        power=np.zeros((T,n))
        start=np.zeros((T,n))
        on=np.zeros((T,n))
        combi=np.zeros(2)
        for t in range(T):
            for i in range(n):
                for r in range(pres):
                    power[t][i] = power[t][i] + vector[t*n*(pres+2)+i*pres+r]
        for t in range(T):
            for i in range(n):
                on[t][i]=vector[t*n*(pres+2)+n*(pres)+i]
                start[t][i]=vector[t*n*(pres+2)+n*(pres+1)+i]
                if power[t][i]>0:
                    for r in range(pres):
                        p[t][i][r]=vector[t*n*(pres+2)+i*pres+r]
        okay, pmincheck, oncheck, startcheck, minupcheck, minwoncheck, errorquality = penaltycheck(vector,dgen,mingen,start,minup,mindown,on,T,n,pres)
        errors.append(errorquality)
        price=costsummation(vector,dgen,mingen,varcost,startcost,T,n,pres)
        combi=[price,okay]
        saveresultsprice.append(combi)

        w=np.transpose(v)
        r=np.matmul(np.matmul(v,Q),w)[0,0]
        combiobjective=[r,okay]
        saveresultsobjective.append(combiobjective)
        if r<cheapestPrice:
            if okay==boolcheck:
                if r<bestobjective:
                    bestobjective=r
                    bestobjectiveAns=vector
                    bestobjectiveprice=price
                if price<bestprice:
                    bestprice=price
                    bestpriceAns=vector
                    bestpriceobj=r
            cheapestPrice=r
            a=v[0]

        combiobjpriceokay=[r,price,errorquality]
        combined.append(np.array(combiobjpriceokay))

    timetosolve=time.process_time() - starttime                                               # calculate time for problem saving
    if graphicsout==True:
        print('time to brute solve the best solution:',timetosolve)

        fig1, ax1 = plt.subplots()                                                # plot energy landscape and costs of all possible solutions

        ax2 = ax1.twinx()
        ax1.plot([k for k in range(len(saveresultsobjective))],[saveresultsobjective[k][0] for k in range(len(saveresultsobjective))],'b-')
        ax2.plot([k for k in range(len(saveresultsprice))],[saveresultsprice[k][0] for k in range(len(saveresultsprice))],'rx')

        ax1.set_xlabel('states', fontsize=20)
        ax1.set_ylabel('objective value', color='b', fontsize=20)
        ax2.set_ylabel('cost value', color='r', fontsize=20)
        plt.show()

    return cheapestPrice , a, combined, errors, bestprice, bestpriceAns, bestpriceobj, bestobjectiveprice, bestobjectiveAns, bestobjective, timetosolve


def MCsteepestdescentUCPqubo(Q,N,dgen,varcost,startcost,mingen,minup,mindown,T,n,pres,bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, boolcheck,graphicsout):

    """
    Variant of the Monte Carlo solver
    """

    starttime = time.process_time()
    v=np.random.randint(2, size=(1, len(Q)))

    result=[]
    current=Q[0][0]
    counter=0
    for m in Q:
        if m[0]==current:
            counter=counter+1
        else:
            current=m[0]
            result.append(counter)
            counter=1
    result.append(counter)
    struktur=result


    currentStart=0
    bestCost=float('inf')
    bestV=[]
    for s in struktur:
        w=np.zeros((1,len(Q)))
        for i in range(currentStart):
            w[0,i]=v[0,i]
        if s==2:
            w[0,currentStart]=v[0,currentStart+1]
            w[0,currentStart+1]=v[0,currentStart]
            currentStart=currentStart+s
        for i in range(currentStart,len(Q)):
            w[0,i]=v[0,i]
        if s!=2:
            currentStart=currentStart+s
        w2=np.transpose(w)
        r=np.matmul(np.matmul(w,Q),w2)[0,0]
        if r<bestCost:
            bestV=w
            bestCost=r

    # Find good solutions and map them back.

    cheapestPrice=float('inf')
    vector=0
    saveresultsprice=[]
    saveresultsobjective=[]
    errors=[]
    combined=[]
    for g in tqdm(range(N)):
        v=np.zeros((1,len(Q)))
        for i in range(len(Q)):
            v[0,i]=round(random.random())
        vector=np.array(v[0]).flatten()
        p = np.zeros((T, n, pres))
        power=np.zeros((T,n))
        start=np.zeros((T,n))
        on=np.zeros((T,n))
        combi=np.zeros(2)
        for t in range(T):
            for i in range(n):
                for r in range(pres):
                    power[t][i] = power[t][i] + vector[t*n*(pres+2)+i*pres+r]
        for t in range(T):
            for i in range(n):
                on[t][i]=vector[t*n*(pres+2)+n*(pres)+i]
                start[t][i]=vector[t*n*(pres+2)+n*(pres+1)+i]
                if power[t][i]>0:
                    for r in range(pres):
                        p[t][i][r]=vector[t*n*(pres+2)+i*pres+r]
        okay, pmincheck, oncheck, startcheck, minupcheck, mindowncheck, errorquality = penaltycheck(vector,dgen,mingen,start,minup,mindown,on,T,n,pres)
        errors.append(errorquality)
        price=costsummation(vector,dgen,mingen,varcost,startcost,T,n,pres)
        combiprice=[price,okay]
        saveresultsprice.append(combiprice)
        w=np.transpose(v)
        r=np.matmul(np.matmul(v,Q),w)[0,0]
        combiobjective=[r,okay]
        saveresultsobjective.append(combiobjective)
        gradientOK=True
        while gradientOK:
            gradR=bestCost
            gradV=bestV

            if gradR>=r:
                gradientOK=False
            else:
                v=gradV
                r=gradR
        if okay==boolcheck:
            if r<bestobjective:
                bestobjective=r
                bestobjectiveAns=vector
                bestobjectiveprice=price
            if price<bestprice:
                bestprice=price
                bestpriceAns=vector
                bestpriceobj=r
        if r<cheapestPrice:
            cheapestPrice=r
            a=v[0]

        combiobjpriceokay=[r,price,errorquality]
        combined.append(np.array(combiobjpriceokay))

    timetosolve=time.process_time() - starttime                                               # calculate time for problem saving
    if graphicsout==True:
        print('time to brute solve the best solution:',timetosolve)

        fig1, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot([k for k in range(len(saveresultsobjective))],[saveresultsobjective[k][0] for k in range(len(saveresultsobjective))],'b-')
        ax2.plot([k for k in range(len(saveresultsprice))],[saveresultsprice[k][0] for k in range(len(saveresultsprice))],'rx')

        ax1.set_xlabel('states', fontsize=20)
        ax1.set_ylabel('objective value', color='b', fontsize=20)
        ax2.set_ylabel('cost value', color='r', fontsize=20)
        plt.show()

    return r, a, combined, errors, bestprice, bestpriceAns, bestpriceobj, bestobjectiveprice, bestobjectiveAns, bestobjective, timetosolve



def SimulatedAnnealingUCPqubo(Q,Num,rounds,dgen,varcost,startcost,mingen,minup,mindown,T,n,pres,bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, boolcheck, Qcost,graphicsout):
    
    """
    Simulated annealing solver using D-Waves "neal" package
    Q: Qubo martix as input
    Num: Number of samples
    rounds: number of subsequent rounds of simulated annealing with initial solutions as input states
    """

    import dimod
    import neal
    import greedy
    import time

    q=Q
    qubo={}
    nonzeros=0
    for i in range(len(q)):
        for j in range(len(q)):
            if abs(q[i][j])>0.00000001:
                qubo[(i,j)]=q[i,j]
                nonzeros=nonzeros+1
    if graphicsout==True:
        print("#nonzeros/#quboEntries",nonzeros,"/",len(Q)**2)
        print("sparsity",1-nonzeros/len(Q)**2)
    bqm = dimod.BQM.from_qubo(qubo)
    starttime = time.process_time()
    SAsamples=neal.SimulatedAnnealingSampler().sample(bqm,num_reads=Num) #sampling with simulated annealing
    timetosolve=time.process_time() - starttime
    for r in range(rounds):  # use samples as initial states for additional runs of simulated annealing, number of runs
        SAsamples=neal.SimulatedAnnealingSampler().sample(bqm,num_reads=Num,initial_states=SAsamples)  # is given by "rounds"

    solver = greedy.SteepestDescentSolver() # now, use steepest descent to lead solution to nearby local minima
    solution=solver.sample(bqm,initial_states=SAsamples)

     # calculate time for problem saving
    if graphicsout==True:
        print('time to find '+str(Num)+' low energy output states with '+str(rounds)+' subsequent runs of simulated annealing: ',timetosolve)

    solobj=[]
    states=[] # manipulating results and post-processing
    for energy, in solution.data(fields=['energy'], sorted_by='energy'):
        solobj.append(energy)
    for state, in solution.data(fields=['sample'], sorted_by='energy'):
        states.append(state)
    solv=dimod.as_samples(states)[0]
    if graphicsout==True:
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx() # plot energy landscape and costs of all possible solutions
        ax1.plot([k for k in range(len(solobj))],solobj,'b-')
        ax2.plot([k for k in range(len(solobj))],[np.matmul(np.matmul(solv[L],Qcost),np.transpose(solv[L])) for L in range(len(solv))],'rx')
        ax2.set_yscale('log')
        ax1.set_xlabel('states', fontsize=20)
        ax1.set_ylabel('objective value', color='b', fontsize=20)
        ax2.set_ylabel('cost value', color='r', fontsize=20)
        plt.show()

    bestSample=solution.first.sample
    Ans=list(bestSample.values())
    price=costsummation(Ans,dgen,mingen,varcost,startcost,T,n,pres)
    combined=[]
    errors=[]
    for L in range(len(solv)):   # prepare for penalty check by "penaltycheck"
        p = np.zeros((T, n, pres))
        power=np.zeros((T,n))
        start=np.zeros((T,n))
        on=np.zeros((T,n))
        combi=np.zeros(2)
        for t in range(T):
            for i in range(n):
                on[t][i]=solv[L][t*n*(pres+2)+n*(pres)+i]
                start[t][i]=solv[L][t*n*(pres+2)+n*(pres+1)+i]
                for r in range(pres):
                    power[t][i] = power[t][i] + solv[L][t*n*(pres+2)+i*pres+r]
                power[t][i]=power[t][i]+solv[L][t*n*(pres+2)+n*pres+i]
        okay, pmincheck, oncheck, startcheck, minupcheck, minwoncheck, errorquality = penaltycheck(solv[L],dgen,mingen,start,minup,mindown,on,T,n,pres)
        errors.append(errorquality)
        price=costsummation(solv[L],dgen,mingen,varcost,startcost,T,n,pres)
        if okay==boolcheck:  # boolcheck=true means only freasible solutions are considered; otherwise unfeasible ones, too
            if solobj[L]<bestobjective:  # find best solution with lowest objective value
                bestobjective=solobj[L]
                bestobjectiveAns=solv[L]
                bestobjectiveprice=price
            if price<bestprice:  # find best solution with lowest costs
                bestprice=price
                bestpriceAns=solv[L]
                bestpriceobj=solobj[L]
        combiobjpriceokay=[solobj[L],np.array(price),np.array(errorquality)]
        combined.append(np.array(combiobjpriceokay))


    return price, Ans, combined, errors, bestprice, bestpriceAns, bestpriceobj, bestobjectiveprice, bestobjectiveAns, bestobjective, timetosolve




def QuantumAnnealingUCPqubo(Q,Num,rounds,DWtoken,dgen,varcost,startcost,mingen,minup,mindown,T,n,pres,bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, boolcheck, Qcost, graphicsout):
    
    """
    Solver that uses quantum annealing machines from D-Wave to calculate a solution
    Q: Qubo martix as input
    Num: Number of samples
    rounds: number of subsequent rounds of simulated annealing with initial solutions as input states
    """

    import dimod
    import neal
    import greedy
    import time

    from dwave.system import LeapHybridSampler
    from dimod import Binary, ConstrainedQuadraticModel, quicksum
    from dimod.binary.binary_quadratic_model import BinaryQuadraticModel, quicksum
    from dimod import ConstrainedQuadraticModel, Integer
    from dwave.system import LeapHybridCQMSampler
    from dwave.system import DWaveSampler


    from dwave.system import DWaveSampler, FixedEmbeddingComposite, EmbeddingComposite
    import dwave.inspector
                
    sampler = EmbeddingComposite(DWaveSampler(token=DWtoken, region="eu-central-1"))  # pass token to sampler, sampler selected to EU region as default
    q=Q
    qubo={}
    nonzeros=0
    for i in range(len(q)):                                                      # build QUBO as dictionary
        for j in range(len(q)):
            if abs(q[i][j])>0.00000001:
                qubo[(i,j)]=q[i,j]
                nonzeros=nonzeros+1
    if graphicsout==True:
        print("#nonzeros/#quboEntries",nonzeros,"/",len(Q)**2)
        print("sparsity",1-nonzeros/len(Q)**2)
    bqm = dimod.BQM.from_qubo(qubo)
    starttime = time.process_time()
    QAsamples=sampler.sample(bqm,num_reads=Num)

    print(time.process_time() - starttime)
    timing_info = QAsamples.info["timing"]
    print(timing_info)
    neal_time = timing_info["qpu_access_time"]
    timetosolve=neal_time * 1e-6
    
    for r in range(rounds):                                                                               # use samples as initial states for additional runs of simulated annealing, number of runs
        QAsamples=neal.SimulatedAnnealingSampler().sample(bqm,num_reads=Num,initial_states=QAsamples)      # is given by "rounds"

    solver = greedy.SteepestDescentSolver()                                                                # use steepest descent to find nearest local minimum    
    solution=solver.sample(bqm,initial_states=QAsamples)

                                                   # calculate time for problem saving
    if graphicsout==True:
        print('time to find '+str(Num)+' low energy output states with '+str(rounds)+' subsequent runs of simulated annealing: ',timetosolve)

    solobj=[]
    states=[]                                                                                                # prepare solutions for post-processing
    for energy, in solution.data(fields=['energy'], sorted_by='energy'):
        solobj.append(energy)
    for state, in solution.data(fields=['sample'], sorted_by='energy'):
        states.append(state)
    solv=dimod.as_samples(states)[0]

    if graphicsout==True:
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()                                                                                            # plot energy landscape and costs of all possible solutions
        ax1.plot([k for k in range(len(solobj))],solobj,'b-')
        ax2.plot([k for k in range(len(solobj))],[np.matmul(np.matmul(solv[L],Qcost),np.transpose(solv[L])) for L in range(len(solv))],'rx')

        ax1.set_xlabel('states', fontsize=20)
        ax1.set_ylabel('objective value', color='b', fontsize=20)
        ax2.set_ylabel('cost value', color='r', fontsize=20)

    bestSample=solution.first.sample
    Ans=list(bestSample.values())
    price=costsummation(Ans,dgen,mingen,varcost,startcost,T,n,pres)
    combined=[]
    errors=[]
    for L in range(len(solv)):                                                                                # prepare for penalty check by "penaltycheck"
        p = np.zeros((T, n, pres))
        power=np.zeros((T,n))
        start=np.zeros((T,n))
        on=np.zeros((T,n))
        combi=np.zeros(2)
        for t in range(T):
            for i in range(n):
                for r in range(pres):
                    power[t][i] = power[t][i] + solv[L][t*n*(pres+2)+i*pres+r]
        for t in range(T):
            for i in range(n):
                on[t][i]=solv[L][t*n*(pres+2)+n*(pres)+i]
                start[t][i]=solv[L][t*n*(pres+2)+n*(pres+1)+i]
                if power[t][i]>0:
                    for r in range(pres):
                        p[t][i][r]=solv[L][t*n*(pres+2)+i*pres+r]
        okay, pmincheck, oncheck, startcheck, minupcheck, minwoncheck, errorquality = penaltycheck(solv[L],dgen,mingen,start,minup,mindown,on,T,n,pres)
        errors.append(errorquality)


        price=costsummation(solv[L],dgen,mingen,varcost,startcost,T,n,pres)
        if okay==boolcheck:                                                                                 # boolcheck=true means only freasible solutions are considered; otherwise unfeasible ones, too
            if solobj[L]<bestobjective:                                                                     # find best solution with lowest objective value
                bestobjective=solobj[L]
                bestobjectiveAns=solv[L]
                bestobjectiveprice=price
            if price<bestprice:                                                                             # find best solution with lowest costs
                bestprice=price
                bestpriceAns=solv[L]
                bestpriceobj=solobj[L]
        combiobjpriceokay=[solobj[L],np.array(price),np.array(errorquality)]
        combined.append(combiobjpriceokay)

    return price, Ans, combined, errors, bestprice, bestpriceAns, bestpriceobj, bestobjectiveprice, bestobjectiveAns, bestobjective, timetosolve

def OptProblem(CostMatrix) -> Problem:

    """
    Azure routine to load matrix to Problem
    """

    terms = []

    for i in range(0, len(CostMatrix)): 
        for j in range(0, len(CostMatrix)): 

            terms.append(
                Term(
                    c = CostMatrix.item((i,j)) ,
                    indices = [i, j]   
                )
            )


    return Problem(name="ucp", problem_type=ProblemType.pubo, terms=terms)


def AzureRoutinePT(workspace,Q,Num,dgen,varcost,startcost,mingen,minup,mindown,T,n,pres,bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, boolcheck, Qcost,graphicsout):

    """
    Microsoft Quantum Inspired Optimization
    Azure quantum-inspired algorithms: Parallel Tempering
    Rephrases the optimization problem as a thermodynamic system and runs multiple copies of a system, randomly initialized, at different temperatures. Then, based on a specific protocol, exchanges configurations at different temperatures to find the optimal configuration.
    Azure workspace need to be loaded
    Q: Qubo martix as input
    Num: Number of samples
    """

    from azure.quantum.optimization import Problem, ProblemType, Term
    from azure.quantum.optimization import ParallelTempering
    
    solver_ = ParallelTempering(workspace)
    Qproblem=OptProblem(Q)

    starttime = time.process_time()
    optresult=solver_.optimize(Qproblem)
    solver_.set_number_of_solutions(Num)

    solobj=[]
    solv=[]
    for i in range(len(optresult['solutions'])):
        y=[]
        y=[item for item in optresult['solutions'][i].values()]
        solobj.append(y[1])
        solv.append(list(y[0].values()))
    
    
    q=Q
    qubo={}
    nonzeros=0
    for i in range(len(q)):
        for j in range(len(q)):
            if abs(q[i][j])>0.00000001:
                qubo[(i,j)]=q[i,j]
                nonzeros=nonzeros+1
    print("#nonzeros/#quboEntries",nonzeros,"/",len(Q)**2)
    print("sparsity",1-nonzeros/len(Q)**2)
    
    timetosolve = time.process_time() - starttime
    print('time to solve QUBO for '+str(Num)+' iterations:',timetosolve)

    if graphicsout==True:
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot([k for k in range(len(solobj))],solobj,'b-')
        ax2.plot([k for k in range(len(solobj))],[np.matmul(np.matmul(solv[L],Qcost),np.transpose(solv[L])) for L in range(len(solv))],'rx')

        ax1.set_xlabel('states', fontsize=20)
        ax1.set_ylabel('objective value', color='b', fontsize=20)
        ax2.set_ylabel('cost value', color='r', fontsize=20)


    Ans=solv[0]
    price5=costsummation(Ans,dgen,mingen,varcost,startcost,T,n,pres)
    combined5=[]
    errors5=[]
    for L in range(len(solv)):
        p5 = np.zeros((T, n, pres))
        power5=np.zeros((T,n))
        start5=np.zeros((T,n))
        on5=np.zeros((T,n))
        combi5=np.zeros(2)
        for t in range(T):
            for i in range(n):
                for r in range(pres):
                    power5[t][i] = power5[t][i] + solv[L][t*n*(pres+2)+i*pres+r]
        for t in range(T):
            for i in range(n):
                on5[t][i]=solv[L][t*n*(pres+2)+n*(pres)+i]
                start5[t][i]=solv[L][t*n*(pres+2)+n*(pres+1)+i]
                if power5[t][i]>0:
                    for r in range(pres):
                        p5[t][i][r]=solv[L][t*n*(pres+2)+i*pres+r]
        okay, pmincheck, oncheck, startcheck, minupcheck, mindowncheck, errorquality = penaltycheck(solv[L],dgen,mingen,start5,minup,mindown,on5,T,n,pres)
        errors5.append(errorquality)
        price=costsummation(solv[L],dgen,mingen,varcost,startcost,T,n,pres)
        if okay==boolcheck:
            if solobj[L]<bestobjective:
                bestobjective=solobj[L]
                bestobjectiveAns=solv[L]
                bestobjectiveprice=price
            if price<bestprice:
                bestprice=price
                bestpriceAns=solv[L]
                bestpriceobj=solobj[L]
        combiobjpriceokay=[solobj[L],price,errorquality]
        combined5.append(combiobjpriceokay)

    return price5, Ans, combined5, errors5, bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, timetosolve

def AzureRoutineQMC(workspace,Q,Num,dgen,varcost,startcost,mingen,minup,mindown,T,n,pres,bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, boolcheck, Qcost,graphicsout):

    """
    Microsoft Quantum Inspired Optimization
    Azure quantum-inspired algorithms: Quantum Monte Carlo
    Similar to Simulated Annealing but the changes are by simulating quantum-tunneling through barriers rather than using thermal energy jumps.
    Azure workspace need to be loaded
    Q: Qubo martix as input
    Num: Number of samples
    """
    
    from azure.quantum.optimization import Problem, ProblemType, Term
    from azure.quantum.optimization import QuantumMonteCarlo
    
    solver_ = QuantumMonteCarlo(workspace)
    Qproblem=OptProblem(Q)
    starttime = time.process_time()
    optresult=solver_.optimize(Qproblem)
    solver_.set_number_of_solutions(Num)
    solobj=[]
    solv=[]
    for i in range(len(optresult['solutions'])):
        y=[]
        y=[item for item in optresult['solutions'][i].values()]
        solobj.append(y[1])
        solv.append(list(y[0].values()))
    
    
    q=Q
    qubo={}
    nonzeros=0
    for i in range(len(q)):
        for j in range(len(q)):
            if abs(q[i][j])>0.00000001:
                qubo[(i,j)]=q[i,j]
                nonzeros=nonzeros+1
    print("#nonzeros/#quboEntries",nonzeros,"/",len(Q)**2)
    print("sparsity",1-nonzeros/len(Q)**2)
    
    timetosolve = time.process_time() - starttime
    print('time to solve QUBO for '+str(Num)+' iterations:',timetosolve)

    if graphicsout==True:
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot([k for k in range(len(solobj))],solobj,'b-')
        ax2.plot([k for k in range(len(solobj))],[np.matmul(np.matmul(solv[L],Qcost),np.transpose(solv[L])) for L in range(len(solv))],'rx')

        ax1.set_xlabel('states', fontsize=20)
        ax1.set_ylabel('objective value', color='b', fontsize=20)
        ax2.set_ylabel('cost value', color='r', fontsize=20)

    Ans=solv[0]
    price6=costsummation(Ans,dgen,mingen,varcost,startcost,T,n,pres)
    combined6=[]
    errors6=[]
    for L in range(len(solv)):
        p6 = np.zeros((T, n, pres))
        power6=np.zeros((T,n))
        start6=np.zeros((T,n))
        on6=np.zeros((T,n))
        combi6=np.zeros(2)
        for t in range(T):
            for i in range(n):
                for r in range(pres):
                    power6[t][i] = power6[t][i] + solv[L][t*n*(pres+2)+i*pres+r]
        for t in range(T):
            for i in range(n):
                on6[t][i]=solv[L][t*n*(pres+2)+n*(pres)+i]
                start6[t][i]=solv[L][t*n*(pres+2)+n*(pres+1)+i]
                if power6[t][i]>0:
                    for r in range(pres):
                        p6[t][i][r]=solv[L][t*n*(pres+2)+i*pres+r]
        okay, pmincheck, oncheck, startcheck, minupcheck, mindowncheck, errorquality = penaltycheck(solv[L],dgen,mingen,start6,minup,mindown,on6,T,n,pres)
        errors6.append(errorquality)
        price=costsummation(solv[L],dgen,mingen,varcost,startcost,T,n,pres)
        if okay==boolcheck:
            if solobj[L]<bestobjective:
                bestobjective=solobj[L]
                bestobjectiveAns=solv[L]
                bestobjectiveprice=price
            if price<bestprice:
                bestprice=price
                bestpriceAns=solv[L]
                bestpriceobj=solobj[L]
        combiobjpriceokay=[solobj[L],price,errorquality]
        combined6.append(combiobjpriceokay)

    return price6, Ans, combined6, errors6, bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, timetosolve

def AzureRoutineSQMC(workspace,Q,Num,dgen,varcost,startcost,mingen,minup,mindown,T,n,pres,bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, boolcheck, Qcost,graphicsout):

    """
    Microsoft Quantum Inspired Optimization
    Azure quantum-inspired algorithms: Substochastic Monte Carlo
    Substochastic Monte Carlo is a diffusion Monte Carlo algorithm inspired by adiabatic quantum computation. It simulates the diffusion of a population of walkers in search space, while walkers are removed or duplicated based on how they perform according to the cost function.
    Azure workspace need to be loaded
    Q: Qubo martix as input
    Num: Number of samples
    """
    
    from azure.quantum.optimization import Problem, ProblemType, Term
    from azure.quantum.optimization import SubstochasticMonteCarlo

    solver_ = SubstochasticMonteCarlo(workspace, seed=48, timeout=10)(workspace)
    Qproblem=OptProblem(Q)
    starttime = time.process_time()
    optresult=solver_.optimize(Qproblem)
    solver_.set_number_of_solutions(Num)
    solobj=[]
    solv=[]
    for i in range(len(optresult['solutions'])):
        y=[]
        y=[item for item in optresult['solutions'][i].values()]
        solobj.append(y[1])
        solv.append(list(y[0].values()))
    
    
    q=Q
    qubo={}
    nonzeros=0
    for i in range(len(q)):
        for j in range(len(q)):
            if abs(q[i][j])>0.00000001:
                qubo[(i,j)]=q[i,j]
                nonzeros=nonzeros+1
    print("#nonzeros/#quboEntries",nonzeros,"/",len(Q)**2)
    print("sparsity",1-nonzeros/len(Q)**2)
    timetosolve = time.process_time() - starttime
    print('time to solve QUBO for '+str(Num)+' iterations:',timetosolve)

    if graphicsout==True:
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot([k for k in range(len(solobj))],solobj,'b-')
        ax2.plot([k for k in range(len(solobj))],[np.matmul(np.matmul(solv[L],Qcost),np.transpose(solv[L])) for L in range(len(solv))],'rx')

        ax1.set_xlabel('states', fontsize=20)
        ax1.set_ylabel('objective value', color='b', fontsize=20)
        ax2.set_ylabel('cost value', color='r', fontsize=20)
    
    Ans=solv[0]
    price7=costsummation(Ans,dgen,mingen,varcost,startcost,T,n,pres)
    combined7=[]
    errors7=[]
    for L in range(len(solv)):
        p7 = np.zeros((T, n, pres))
        power7=np.zeros((T,n))
        start7=np.zeros((T,n))
        on7=np.zeros((T,n))
        combi7=np.zeros(2)
        for t in range(T):
            for i in range(n):
                for r in range(pres):
                    power7[t][i] = power7[t][i] + solv[L][t*n*(pres+2)+i*pres+r]
        for t in range(T):
            for i in range(n):
                on7[t][i]=solv[L][t*n*(pres+2)+n*(pres)+i]
                start7[t][i]=solv[L][t*n*(pres+2)+n*(pres+1)+i]
                if power7[t][i]>0:
                    for r in range(pres):
                        p7[t][i][r]=solv[L][t*n*(pres+2)+i*pres+r]
        okay, pmincheck, oncheck, startcheck, minupcheck, mindowncheck, errorquality = penaltycheck(solv[L],dgen,mingen,start7,minup,mindown,on7,T,n,pres)
        errors7.append(errorquality)
        price=costsummation(solv[L],dgen,mingen,varcost,startcost,T,n,pres)
        if okay==boolcheck:
            if solobj[L]<bestobjective:
                bestobjective=solobj[L]
                bestobjectiveAns=solv[L]
                bestobjectiveprice=price
            if price<bestprice:
                bestprice=price
                bestpriceAns=solv[L]
                bestpriceobj=solobj[L]
        combiobjpriceokay=[solobj[L],price,errorquality]
        combined7.append(combiobjpriceokay)

    return price7, Ans, combined7, errors7, bestprice, bestpriceAns, bestobjective, bestobjectiveAns, bestobjectiveprice, bestpriceobj, timetosolve

def buildUCPqubo(autoset,n,pres,T,d,dgen,Clist,varcost,startcost,minup,mindown,mingen,maxgen):

    """
    Build the QUBO for the unit commitment problem as outlined in the following paper:
    M.C. Braun, T. Decker, N. Hegemann, S.F. Kerstan, F. Lorenz, "Towards optimization under uncertainty for fundamental models in energy markets using quantum computers", arXiv:2301.01108
    
    Input:
    List of power unit parameters, precision, time steps, demand, costs, min up/down min/max supply parameters
    """

    qubostart = time.process_time()


    #calculating auxiliary variables
    dim=n*(pres+2)*T     #dimension of QUBO (size)
    dimt=n*(pres+2)      #size of one time step


    #set penalty strengths
    kappa=n*max(np.add(varcost*np.subtract(maxgen,mingen),startcost))

    if autoset[0]==1:                        #set penalty strengths
        A=1
        B=((A*kappa**2)/((np.min(dgen))**2))*2
        C=1e2*kappa**2
        C2=C
        D=1e4*kappa**2
        E=1e6*kappa**2
    else:
        A=autoset[1]
        B=autoset[2]
        C=autoset[3]
        C2=autoset[4]
        D=autoset[5]
        E=autoset[6]


    #building QUBO_{cost}
    Qcost = np.zeros((dim,dim))
    for t in range(T):
        for i in range(n):
            Qcost[t*dimt+n*pres+i][t*dimt+n*pres+i] =Qcost[t*dimt+n*pres+i][t*dimt+n*pres+i]+ A*varcost[i]*mingen[i]
            Qcost[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i] =Qcost[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]+ A*startcost[i]
            for r in range(pres):
                Qcost[t*dimt+i*pres+r][t*dimt+i*pres+r] =Qcost[t*dimt+i*pres+r][t*dimt+i*pres+r]+ A*varcost[i]*dgen[i][r]


    #building QUBO_{quadratic costs}
    Qcostq = np.zeros((dim,dim))
    for t1 in range(T):
        for t2 in range(T):
            for i1 in range(n):
                for i2 in range(n):
                    Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*pres+i2] =Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*pres+i2] + A*varcost[i1]*varcost[i2]*mingen[i1]*mingen[i2]
                    Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*(pres+1)+i2]=Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*(pres+1)+i2] + 2*A*varcost[i1]*mingen[i1]*startcost[i2]
                    Qcostq[t1*dimt+n*(pres+1)+i1][t2*dimt+n*(pres+1)+i2]=Qcostq[t1*dimt+n*(pres+1)+i1][t2*dimt+n*(pres+1)+i2] + A*startcost[i1]*startcost[i2]
                    for r1 in range(pres):
                        Qcostq[t1*dimt+n*pres+i1][t2*dimt+i2*pres+r1] =Qcostq[t1*dimt+n*pres+i1][t2*dimt+i2*pres+r1]+ 2*A*varcost[i1]*mingen[i1]*dgen[i2][r1]
                        Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*(pres+1)+i2] =Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*(pres+1)+i2]+ 2*A*varcost[i1]*dgen[i1][r1]*startcost[i2]
                        for r2 in range(pres):
                            Qcostq[t1*dimt+i1*pres+r1][t2*dimt+i2*pres+r2]=Qcostq[t1*dimt+i1*pres+r1][t2*dimt+i2*pres+r2]+A*varcost[i1]*varcost[i2]*dgen[i1][r1]*dgen[i2][r2]


    #building QUBO_{demand}
    Qdemand = np.zeros((dim,dim))
    for t in range(T):
        for i in range(n):
            Qdemand[t*dimt+n*pres+i][t*dimt+n*pres+i] =Qdemand[t*dimt+n*pres+i][t*dimt+n*pres+i]-2*B*d[t]*mingen[i]
            for j in range(n):
                Qdemand[t*dimt+n*pres+i][t*dimt+n*pres+j] =Qdemand[t*dimt+n*pres+i][t*dimt+n*pres+j]+ B*mingen[i]*mingen[j]
                for r in range(pres):
                    Qdemand[t*dimt+n*pres+i][t*dimt+j*pres+r] =Qdemand[t*dimt+n*pres+i][t*dimt+j*pres+r]+ 2*B*mingen[i]*dgen[j][r]
                    for r2 in range(pres):
                        Qdemand[t*dimt+i*pres+r][t*dimt+j*pres+r2] =Qdemand[t*dimt+i*pres+r][t*dimt+j*pres+r2]+ B*dgen[i][r]*dgen[j][r2]
            for r in range(pres):
                Qdemand[t*dimt+i*pres+r][t*dimt+i*pres+r] =Qdemand[t*dimt+i*pres+r][t*dimt+i*pres+r]-2*B*d[t]*dgen[i][r]


    #building QUBO_{minup}
    Qtmin = np.zeros((dim,dim))
    for t in range(T):
        for i in range(n):
            if t+minup[i]>T:
                endt=T
            else:
                endt=t+minup[i]
            Qtmin[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]=Qtmin[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]+C*minup[i]
            for tau in range(t,endt):
                Qtmin[t*dimt+n*(pres+1)+i][tau*dimt+n*pres+i]=Qtmin[t*dimt+n*(pres+1)+i][tau*dimt+n*pres+i]-C

    #building QUBO_{mindown}
    Qtdown = np.zeros((dim,dim))
    for t in range(1,T):
        for i in range(n):
            if t+mindown[i]>T:
                endt=T
            else:
                endt=t+mindown[i]
            Qtdown[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]=Qtdown[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]+C2*mindown[i]
            for tau in range(t,endt):
                Qtdown[(t-1)*dimt+n*pres+i][tau*dimt+n*pres+i]=Qtdown[(t-1)*dimt+n*pres+i][tau*dimt+n*pres+i]+C2
                Qtdown[t*dimt+n*pres+i][tau*dimt+n*pres+i]=Qtdown[t*dimt+n*pres+i][tau*dimt+n*pres+i]-C2

    #building QUBO_{interrelate1}
    Qint1 = np.zeros((dim,dim))
    for t in range(T):
        for i in range(n):
            for r in range(pres):
                Qint1[t*dimt+i*pres+r][t*dimt+i*pres+r]=Qint1[t*dimt+i*pres+r][t*dimt+i*pres+r]+D
                Qint1[t*dimt+i*pres+r][t*dimt+n*pres+i]=Qint1[t*dimt+i*pres+r][t*dimt+n*pres+i]-D

    #building QUBO_{interrelate2}
    Qint2 = np.zeros((dim,dim))
    for t in range(T):
        for i in range(n):
            if t>0:
                Qint2[t*dimt+n*pres+i][t*dimt+n*pres+i]=Qint2[t*dimt+n*pres+i][t*dimt+n*pres+i]+E
                Qint2[t*dimt+n*pres+i][t*dimt+n*(pres+1)+i]=Qint2[t*dimt+n*pres+i][t*dimt+n*(pres+1)+i]-2*E
                Qint2[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]=Qint2[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]+E
                Qint2[(t-1)*dimt+n*pres+i][t*dimt+n*(pres+1)+i]=Qint2[(t-1)*dimt+n*pres+i][t*dimt+n*(pres+1)+i]+E
                Qint2[(t-1)*dimt+n*pres+i][t*dimt+n*pres+i]=Qint2[(t-1)*dimt+n*pres+i][t*dimt+n*pres+i]-E
            else:
                Qint2[t*dimt+n*pres+i][t*dimt+n*pres+i]=Qint2[t*dimt+n*pres+i][t*dimt+n*pres+i]+E
                Qint2[t*dimt+n*pres+i][t*dimt+n*(pres+1)+i]=Qint2[t*dimt+n*pres+i][t*dimt+n*(pres+1)+i]-2*E
                Qint2[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]=Qint2[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]+E

    #if graphicsout==True:
    print('time to build QUBO:',time.process_time() - qubostart)                    #print Build time for QUBO

    return   (Qcostq + Qdemand + Qtmin + Qtdown + Qint1 + Qint2), Qcost


def buildSUCPqubo(autoset,n,pres,T,d,dgen,Clist,varcost,startcost,minup,mindown,mingen,maxgen,pdRE,pdD,expd):

    """
    Build the relaxed QUBO for the unit commitment problem including uncertainty as outlined in the following paper:
    M.C. Braun, T. Decker, N. Hegemann, S.F. Kerstan, F. Lorenz, "Towards optimization under uncertainty for fundamental models in energy markets using quantum computers", arXiv:2301.01108
    
    Input:
    List of power unit parameters, precision, time steps, demand, costs, min up/down min/max supply parameters, list of demand and supply including probabilities from renewables
    """

    qubostart = time.process_time()


    #calculating auxiliary variables
    REres=len(pdRE[0][0])
    Dres=len(pdD[0])
    lREnum=len(pdRE[0])
    lREt=len(pdRE[0])*len(pdRE[0][0])
    lRE=T*lREnum*REres
    lD =T*Dres

    dim=n*(pres+2)*T     #dimension of QUBO (size) without stochastic part
    dimt=n*(pres+2)      #size of one time step
    dimc=dim+lRE+lD      #complete dimension of QUBO including stochastic part

    #set penalty strengths
    kappa=n*max(np.add(varcost*np.subtract(maxgen,mingen),startcost))

    if autoset[0]==1:                        #set penalty strengths
        A=1
        B=((A*kappa**2)/((np.min(dgen))**2))*2
        C=1e2*kappa**2
        C2=C
        D=1e4*kappa**2
        E=1e6*kappa**2
        F=1e-2*B
        G=1e8*((A*kappa**2)/((np.min(dgen))**2))*2
        H=1e8*((A*kappa**2)/((np.min(dgen))**2))*2
    else:
        A=autoset[1]
        B=autoset[2]
        C=autoset[3]
        C2=autoset[4]
        D=autoset[5]
        E=autoset[6]
        F=autoset[7]
        G=autoset[8]
        H=autoset[9]


    #building QUBO_{cost}
    Qcost = np.zeros((dimc,dimc))
    for t in range(T):
        for i in range(n):
            Qcost[t*dimt+n*pres+i][t*dimt+n*pres+i] =Qcost[t*dimt+n*pres+i][t*dimt+n*pres+i]+ A*varcost[i]*mingen[i]
            Qcost[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i] =Qcost[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]+ A*startcost[i]
            for r in range(pres):
                Qcost[t*dimt+i*pres+r][t*dimt+i*pres+r] =Qcost[t*dimt+i*pres+r][t*dimt+i*pres+r]+ A*varcost[i]*dgen[i][r]


    #building QUBO_{quadratic costs}
    Qcostq = np.zeros((dimc,dimc))
    for t1 in range(T):
        for t2 in range(T):
            for i1 in range(n):
                for i2 in range(n):
                    Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*pres+i2] =Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*pres+i2] + A*varcost[i1]*varcost[i2]*mingen[i1]*mingen[i2]
                    Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*(pres+1)+i2]=Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*(pres+1)+i2] + 2*A*varcost[i1]*mingen[i1]*startcost[i2]
                    Qcostq[t1*dimt+n*(pres+1)+i1][t2*dimt+n*(pres+1)+i2]=Qcostq[t1*dimt+n*(pres+1)+i1][t2*dimt+n*(pres+1)+i2] + A*startcost[i1]*startcost[i2]
                    for r1 in range(pres):
                        Qcostq[t1*dimt+n*pres+i1][t2*dimt+i2*pres+r1] =Qcostq[t1*dimt+n*pres+i1][t2*dimt+i2*pres+r1]+ 2*A*varcost[i1]*mingen[i1]*dgen[i2][r1]
                        Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*(pres+1)+i2] =Qcostq[t1*dimt+n*pres+i1][t2*dimt+n*(pres+1)+i2]+ 2*A*varcost[i1]*dgen[i1][r1]*startcost[i2]
                        for r2 in range(pres):
                            Qcostq[t1*dimt+i1*pres+r1][t2*dimt+i2*pres+r2]=Qcostq[t1*dimt+i1*pres+r1][t2*dimt+i2*pres+r2]+A*varcost[i1]*varcost[i2]*dgen[i1][r1]*dgen[i2][r2]


    #building QUBO_{demand}
    Qstdemand = np.zeros((dimc,dimc))
    for t in range(T):
        for i in range(n):
            for ii in range(n):
                Qstdemand[t*dimt+n*pres+i][t*dimt+n*pres+ii]=Qstdemand[t*dimt+n*pres+i][t*dimt+n*pres+ii]+B*mingen[i]*mingen[ii]
                for r1 in range(pres):
                    Qstdemand[t*dimt+n*pres+i][t*dimt+ii*pres+r1]=Qstdemand[t*dimt+n*pres+i][t*dimt+ii*pres+r1]+2*B*mingen[i]*dgen[ii][r1]
                    for r2 in range(pres):
                        Qstdemand[t*dimt+i*pres+r1][t*dimt+ii*pres+r2]=Qstdemand[t*dimt+i*pres+r1][t*dimt+ii*pres+r2]+B*dgen[i][r1]*dgen[ii][r2]
            for l in range(Dres):
                Qstdemand[t*dimt+n*pres+i][dim+lRE+t*Dres+l]=Qstdemand[t*dimt+n*pres+i][dim+lRE+t*Dres+l]-2*B*mingen[i]*pdD[t][l]
            for j in range(lREnum):
                for s in range(REres):
                    Qstdemand[t*dimt+n*pres+i][dim+t*lREt+j*REres+s]=Qstdemand[t*dimt+n*pres+i][dim+t*lREt+j*REres+s]+2*B*mingen[i]*pdRE[t][j][s]
            for r in range(pres):
                for l in range(Dres):
                    Qstdemand[t*dimt+i*pres+r][dim+lRE+t*Dres+l]=Qstdemand[t*dimt+i*pres+r][dim+lRE+t*Dres+l]-2*B*dgen[i][r]*pdD[t][l]
                for j in range(lREnum):
                    for s in range(REres):
                        Qstdemand[t*dimt+i*pres+r][dim+t*lREt+j*REres+s]=Qstdemand[t*dimt+i*pres+r][dim+t*lREt+j*REres+s]+2*B*dgen[i][r]*pdRE[t][j][s]
        for j in range(lREnum):
            for jj in range(lREnum):
                for s1 in range(REres):
                    for s2 in range(REres):
                        Qstdemand[dim+t*lREt+j*REres+s1][dim+t*lREt+jj*REres+s2]=Qstdemand[dim+t*lREt+j*REres+s1][dim+t*lREt+jj*REres+s2]+B*pdRE[t][j][s1]*pdRE[t][jj][s2]
        for l1 in range(Dres):
            for l2 in range(Dres):
                Qstdemand[dim+lRE+t*Dres+l1][dim+lRE+t*Dres+l2]=Qstdemand[dim+lRE+t*Dres+l1][dim+lRE+t*Dres+l2]+B*pdD[t][l1]*pdD[t][l2]
            for j in range(lREnum):
                for s in range(REres):
                    Qstdemand[dim+lRE+t*Dres+l1][dim+t*lREt+j*REres+s]=Qstdemand[dim+lRE+t*Dres+l1][dim+t*lREt+j*REres+s]-2*B*pdD[t][l1]*pdRE[t][j][s]


    #building QUBO_{minup}
    Qtmin = np.zeros((dimc,dimc))
    for t in range(T):
        for i in range(n):
            if t+minup[i]>T:
                endt=T
            else:
                endt=t+minup[i]
            Qtmin[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]=Qtmin[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]+C*minup[i]
            for tau in range(t,endt):
                Qtmin[t*dimt+n*(pres+1)+i][tau*dimt+n*pres+i]=Qtmin[t*dimt+n*(pres+1)+i][tau*dimt+n*pres+i]-C

    #building QUBO_{mindown}
    Qtdown = np.zeros((dimc,dimc))
    for t in range(1,T):
        for i in range(n):
            if t+mindown[i]>T:
                endt=T
            else:
                endt=t+mindown[i]
            Qtdown[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]=Qtdown[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]+C2*mindown[i]
            for tau in range(t,endt):
                Qtdown[(t-1)*dimt+n*pres+i][tau*dimt+n*pres+i]=Qtdown[(t-1)*dimt+n*pres+i][tau*dimt+n*pres+i]+C2
                Qtdown[t*dimt+n*pres+i][tau*dimt+n*pres+i]=Qtdown[t*dimt+n*pres+i][tau*dimt+n*pres+i]-C2

    #building QUBO_{interrelate1}
    Qint1 = np.zeros((dimc,dimc))
    for t in range(T):
        for i in range(n):
            for r in range(pres):
                Qint1[t*dimt+i*pres+r][t*dimt+i*pres+r]=Qint1[t*dimt+i*pres+r][t*dimt+i*pres+r]+D
                Qint1[t*dimt+i*pres+r][t*dimt+n*pres+i]=Qint1[t*dimt+i*pres+r][t*dimt+n*pres+i]-D

    #building QUBO_{interrelate2}
    Qint2 = np.zeros((dimc,dimc))
    for t in range(T):
        for i in range(n):
            if t>0:
                Qint2[t*dimt+n*pres+i][t*dimt+n*pres+i]=Qint2[t*dimt+n*pres+i][t*dimt+n*pres+i]+E
                Qint2[t*dimt+n*pres+i][t*dimt+n*(pres+1)+i]=Qint2[t*dimt+n*pres+i][t*dimt+n*(pres+1)+i]-2*E
                Qint2[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]=Qint2[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]+E
                Qint2[(t-1)*dimt+n*pres+i][t*dimt+n*(pres+1)+i]=Qint2[(t-1)*dimt+n*pres+i][t*dimt+n*(pres+1)+i]+E
                Qint2[(t-1)*dimt+n*pres+i][t*dimt+n*pres+i]=Qint2[(t-1)*dimt+n*pres+i][t*dimt+n*pres+i]-E
            else:
                Qint2[t*dimt+n*pres+i][t*dimt+n*pres+i]=Qint2[t*dimt+n*pres+i][t*dimt+n*pres+i]+E
                Qint2[t*dimt+n*pres+i][t*dimt+n*(pres+1)+i]=Qint2[t*dimt+n*pres+i][t*dimt+n*(pres+1)+i]-2*E
                Qint2[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]=Qint2[t*dimt+n*(pres+1)+i][t*dimt+n*(pres+1)+i]+E

    #building QUBO_{minimal_d_variance}
    Qmdv=np.zeros((dimc,dimc))
    for t in range(T):
        for t2 in range(T):
            for l1 in range(Dres):
                Qmdv[dim+lRE+t*Dres+l1][dim+lRE+t*Dres+l1]=Qmdv[dim+lRE+t*Dres+l1][dim+lRE+t*Dres+l1]-2*F*pdD[t][l1]*expd[t2]
                for l2 in range(Dres):
                    Qmdv[dim+lRE+t*Dres+l1][dim+lRE+t2*Dres+l2]=Qmdv[dim+lRE+t*Dres+l1][dim+lRE+t2*Dres+l2]+F*pdD[t][l1]*pdD[t2][l2]
                for j in range(lREnum):
                    for s in range(REres):
                        Qmdv[dim+lRE+t*Dres+l1][dim+t2*lREt+j*REres+s]=Qmdv[dim+lRE+t*Dres+l1][dim+t2*lREt+j*REres+s]-2*F*pdD[t][l1]*pdRE[t2][j][s]
            for j in range(lREnum):
                for jj in range(lREnum):
                    for s1 in range(REres):
                        for s2 in range(REres):
                            Qmdv[dim+t*lREt+j*REres+s1][dim+t2*lREt+jj*REres+s2]=Qmdv[dim+t*lREt+j*REres+s1][dim+t2*lREt+jj*REres+s2]+F*pdRE[t1][j][s1]*pdRE[t2][jj][s2]
            for j in range(lREnum):
                    for s in range(REres):
                        Qmdv[dim+t*lREt+j*REres+s][dim+t*lREt+j*REres+s]=Qmdv[dim+t*lREt+j*REres+s][dim+t*lREt+j*REres+s]+2*F*pdRE[t][j][s]*expd[t2]

    #build QUBO_{choose a unambiguous scenario}
    Qunique=np.zeros((dimc,dimc))
    for t in range(T):
        for j in range(lREnum):
            for s in range(REres):
                Qunique[dim+t*lREt+j*REres+s][dim+t*lREt+j*REres+s]=Qunique[dim+t*lREt+j*REres+s][dim+t*lREt+j*REres+s]-2*H
                for s2 in range(REres):
                    Qunique[dim+t*lREt+j*REres+s][dim+t*lREt+j*REres+s2]=Qunique[dim+t*lREt+j*REres+s][dim+t*lREt+j*REres+s2]+H
        for l in range(Dres):
            Qunique[dim+lRE+t*Dres+l][dim+lRE+t*Dres+l]=Qunique[dim+lRE+t*Dres+l][dim+lRE+t*Dres+l]-2*G
            for l2 in range(Dres):
                Qunique[dim+lRE+t*Dres+l][dim+lRE+t*Dres+l2]=Qunique[dim+lRE+t*Dres+l][dim+lRE+t*Dres+l2]+G

    #if graphicsout==True:
    print('time to build QUBO:',time.process_time() - qubostart)                    #print Build time for QUBO

    return   (Qcostq + Qstdemand + Qtmin + Qtdown + Qint1 + Qint2 + Qmdv + Qunique), Qcost