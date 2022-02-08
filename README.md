# pygrnd 
is a libary of various quantum algorithms written by the Team JoS QUANTUM GmbH to support the development of Applications for quantum computing in Finance, Insurance and Energy.

## Install pygrnd

Pre-required: Qiskit Version 'qiskit': '0.32.1'

pip install qiskit
pip install numpy
pip install dimod
pip install greedy

got to /[yourpath]/pygrnd

pip install .

## Tutorials

## Amplitude Estimation without Phase Estimation

- We have an operator A and a set of good results that have probability a in total.
- We construct the Grover oracle G for A.
- We construct a circuit with m calls of G in addition to one call of A and we run the circuit N times. We obtain h good results.
- For different values of m we repeat this procedure.
- We obtain vectors vectorN, vectorM and vectorH.
- Use these parameters to guess the value a with maximum likelihood method.
- The method uses a gradient descent with a random starting point and a stepSize=0.01 as default step size.
- The method starts with rounds=10 as default number of random starting points.

Example:

    from pygrnd.qc.MaximumLikelihoodForQAEwoPE import loopGradientOptimizerVector
    vectorN=[30,30,30]
    vectorM=[0,1,2]
    vectorH=[21, 1, 29]
    bestTheta,bestProb=loopGradientOptimizerVector(vectorN, vectorM, vectorH, rounds=10, stepSize=0.01)
    print("best guess theta=",bestTheta)
    print("best guess prob=",bestProb)

## Pattern-based Circuit optimizer

#### Background: Patterns

A pattern is an abstract representation of a sequence of quantum gates that corresponds to the identity. The gates are denoted by a string, e.g. H or X, and the parameters are variables or concrete values. The gates act on qubits that are denoted by numbers only.

An example for a pattern:

    [['CNOT', {}, [0, 1]], ['RX', {'theta': 1.571}, [1]], ['H', {}, [0]], ['RXX', {'theta': -1.571}, [0, 1]], ['H', {}, [0]], ['S', {}, [0]]

For instance, ['RXX', {'theta': -1.571}, [0, 1]] denotes a RXX gate with parameter theta set to -1.571 that acts on qubit 0 and 1.

Parameters can also be variables. The idea is that we would like to replace two consecutive rotations by a single one, i.e., we would like to replace R(a)R(b) by the rotation R(a+b). For two variables, we encode variable a by the list [1,0] and variable b by [0,1]. The sum is then [1,1], i.e. we have two basis elements and we allow linear combinations.

This can be rewritten in the following pattern that represents the identity:

    [ ['P',{'lambda':[0,1]},[0]], ['P',{'lambda':[1,0]},[0]],['P',{'lambda':[-1,-1]},[0]]]

When doing calculations with qiskit, the variables [1,0] and [0,1] are replaced by the concrete values 0.1 and 0.6 internally. As all reductions are checked this simplification should not raise any problems.

#### Background: Reduction with a pattern

The pattern-based simplification works like follows:
- Take the original circuit and create a list of all possible candidates. A candidate is a list of gates that can be considered as a single block.
- Take the list of patterns and try to match a part of the pattern with the candidate. The pattern is longer than the candidate.
- If the variables in the pattern can be matched with the concrete values in the candidate, then we consider the remaining part of the pattern.
- If the remaining part of the pattern has lower costs than the replaced candidate, then we replace it by its inverse.

For instance, the candidate [ ['P',{'lambda':0.25},[0]], ['P',{'lambda':0.35},[0]]] matches the first two gates of the pattern

    [ ['P',{'lambda':[0,1]},[0]], ['P',{'lambda':[1,0]},[0]],['P',{'lambda':[-1,-1]},[0]]]

with the variable b (represented by [0,1]) set to 0.35 and variable a (represented by [1,0]) set to 0.25. The inverse of the remaining part of the pattern
is then [ ['P',{'lambda':0.6},[0]]]. Therefore, the P gates with parameters 0.25 and 0.35 can be replaced by a single P gate with the parameter 0.6.

#### Pattern Generation

**Syntax**

    getIdentities(totalQubits, numberGates, prefixPattern, gates, params, qubits)

**Parameters**
- totalQubits: Number of qubits for our patterns, 2 or 3 work well
- numberGates: Number of gates on the qubits, up to 5 works well
- prefixPattern: This pattern is always attached to the front of a candidate. Useful if we want to replace gates with gates on less qubits.
- gates: A list of strings. Each string denotes a gate type, e.g. X or H.
- params: Same length as parameter gates. Each element is a dictionary for parameters for the corresponding gate, e.g. {'lambda':0.2} for a P gate
- qubits: Same length as parameter gates. The number of qubits for the corresponding gates, e.g. 1 for an H gate

**Supported gates**
H, Z, SWAP, CNOT, CCX, CZ, S, Sdg, T, Tdg, SX, SXdg, P, RXX, GPI, GPI2, GZ, CP, RX, RY, RZ

**Examples**

#### Pattern Reduction

**Syntax**

    reduceCircuitByPattern(qc, consideredQubits, allPatterns, costPattern)

**Parameters**
- qc: Quantum circuit to be reduced
- consideredQubits: Restrict candidate creation to this number of qubits.
- allPatterns: The pattern database we want to use.
- costPatterns: A function that is used to compare the cost of patterns. The method uses this function to reduce the costs.

**Supported gates**
H, Z, SWAP, CNOT, CCX, CZ, S, Sdg, T, Tdg, SX, SXdg, P, RXX, GPI, GPI2, GZ, CP, RX, RY, RZ

**Examples**



## Optimizer

Implementations of different classical QUBO (Quadratic Unconstrained Binary Optimization) solver for benchmarks.
Usually a QUBO is defined as a numpy array or dict.


#### Monte Carlo random search

- randomly generate solution vector
- calculation of matrix product x^T Q x
- iterate N times

MonteCarloSolver.py

**Syntax**
    r,a=MCfullsolver(Q,N)

**Parameters**
- Q: QUBO matrix as numpy array
- N: number of runs
- r: solution value
- a: solution vetor

**Examples**
    from pygrnd.optimize.MonteCarloSolver import *
    Q = np.array([[10,-3,-4,-6],[-3,4,-2,-3],[-4,-2,6,-5],[-6,-3,-5,12]])
    N=10
    r,a=MCfullsolver(Q,N)
    print(r,a)

#### Monte Carlo gradient search

- randomly generate solution vector
- map to graph structure 
- calculation of matrix product x^T Q x
- iterate N times

MonteCarloGradientSearch.py

**Syntax**
    r,a=MCfullsolver(Q,N)

**Parameters**
- Q: QUBO matrix as numpy array
- N: number of runs
- r: solution value
- a: solution vetor

**Examples**
    from pygrnd.optimize.MonteCarloGradientSearch import *
    Q = np.array([[10,-3,-4,-6],[-3,4,-2,-3],[-4,-2,6,-5],[-6,-3,-5,12]])
    N=10
    r,a=MCgradientSearch(Q,10)
    print(r,a)

#### Quantum Approximate Optimization Algorithm (QAOA)

- input QUBO matrix as numpy array
- build cost Hamiltonian U(gamma) using mapping x --> (x - 1) /2 (matrixConvertInv(m))
- build mixer Hamiltonian U(beta)
- construct one layer circuit with initial random beta, gamma values
- execute on backend (default: Aer.get_backend('qasm_simulator'))
- derive max count
- calculate expectation value
- return to optimizer
- append layer
- optimize betas and gammas using Nelder-Mead scipy optimizer
- return best betas and gammas, circuit, objective value

qaoa.py

**Syntax**

Helper functions:
num2bin(x,r)
allCombinations(n)
counts2probs(counts)
fidelityCounts(countsP, countsQ)
maxString(counts)
eval_solution(x,m)
matrixConvertInv(m)
addGates(qr,qc,m1,gamma)
qaoaLandscape(m,n,Nshots)

Main functions:
vec, counts, obj, qc, prob = qaoaExp(m0,beta,gamma,Nshots,backend = Aer.get_backend('qasm_simulator'))
obj = multiLayerqaoaExp(m,betas,gammas,Nshots,backend = Aer.get_backend('qasm_simulator'))
vec, counts, obj, prob, qc = multiLayerqaoa(m,betas,gammas,Nshots,backend = Aer.get_backend('qasm_simulator'))

Full workflow:
vec, counts, obj, prob, qc, res1, res2, bestBetas, bestGammas = QAOAoptimize(m,layer,Nshots,backend = Aer.get_backend('qasm_simulator'))


**Parameters**
    - m: QUBO matrix
    - layer: number of layer

**Examples**
layer = 10
Nshots = 10000
vec, counts, obj, prob, qc, res2 = optimise(xxs, layer,Nshots)

