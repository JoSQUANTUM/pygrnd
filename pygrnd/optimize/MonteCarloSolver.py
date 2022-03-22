
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
