
# Solve QUBOs by randomly multiplying (0,1) vectors and checking costs
# QUBO Q, N runs

def MCsolver(Q,N):

    import random
    from tqdm import tqdm
    # Find good solutions and map them back.

    cheapestPrice=float('inf')
    vector=0

    #while True:
    for i in tqdm(range(N)):
        v=np.zeros((1,len(Q)))
        for i in range(len(Q)):
            v[0,i]=round(random.random())
        w=np.transpose(v)
        r=np.matmul(np.matmul(v,Q),w)[0,0]
        if r<cheapestPrice:
            cheapestPrice=r
            a=v[0]

    return r,a
    
# gradient search solver to solve QUBOs that are represented as numpy arrays
# QUBO Q, N runs


def MCgradientSearch(Q,N):
    
    import random
    import numpy as np
    from tqdm import tqdm
    
    v=np.random.randint(2, size=(1, len(Q)))
    
    #def strukturGraph(nodes):
    result=[]
    current=Q[0][0]
    counter=0
    for n in Q:
        if n[0]==current:
            counter=counter+1
        else:
            current=n[0]
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

    #while True:
    for i in tqdm(range(N)):
        v=np.zeros((1,len(Q)))
        for i in range(len(Q)):
            v[0,i]=round(random.random())
        w=np.transpose(v)
        r=np.matmul(np.matmul(v,Q),w)[0,0]
        gradientOK=True
        while gradientOK:
            gradR=bestCost
            gradV=bestV

            if gradR>=r:
                gradientOK=False
            else:
                v=gradV
                r=gradR
        if r<cheapestPrice:
            cheapestPrice=r
            a=v[0]
            print(r,a,list(map(round,a)))

    return r, a
