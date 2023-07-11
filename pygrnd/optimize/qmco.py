# Mathematics and Data
import math
import numpy as np                   # Scientific computing
np.set_printoptions(precision=3, suppress=True)
import pandas as pd                  # Data frames

# Utilities
from tqdm import tqdm                # Progress bars
from IPython.display import display  # Display pandas dataframes
import os                            # Operating system interfaces (e.g. write to files)
import time                          # Time access and conversions
import itertools, operator           # Generate objective function weights for weighed sum scalarization

# Plots and visualizations
import matplotlib.pyplot as plt      # Library for data visualization
import seaborn as sns                # Library for data visualization
import colorcet as cc                # Better colormaps
import matplotlib.colors
import matplotlib.gridspec           # Grids of subplots

# Quantum Computing (D-Wave packages)
import dimod, neal, greedy
from dwave.system import DWaveSampler, EmbeddingComposite

# Classical Optimization
import scipy.optimize as sco


#------------
# Basic QUBO
#------------

def __qubo_obj(A, b, c, v, vvᵀ):
    """Returns the QUBO matrix for the objective "min wᵀAw + bᵀw + c", where w ∈ [0,1]ⁿ,
    given the scaling factors stored as `v` and `vvᵀ`."""
    res = 0
    if not (np.isscalar(A) and A == 0):
        res += np.kron(A, vvᵀ) 
    if not (np.isscalar(b) and b == 0):
        res += np.diag(np.kron(b, v))
    return res

def __qubo_cstr(A, b, c, v, vvᵀ): # A is not used
    """Returns the QUBO matrix for the constraint "bᵀw + c = 0", where w ∈ [0,1]ⁿ,
    given the scaling factors stored as `v` and `vvᵀ`."""
    if np.isscalar(b) and b == 0:
        return 0
    return __qubo_obj(np.outer(b, b), 2*c*np.array(b), 0, v, vvᵀ)

def __calc_obj(A, b, c, w):
    """For w ∈ [0,1]ⁿ, returns the value of wᵀAw + bᵀw + c."""
    res = c
    if not (np.isscalar(A) and A == 0):
        res += (w.T @ A @ w)
    if not (np.isscalar(b) and b == 0):
        res += np.dot(b, w)
    return 0 if np.isclose(res, 0) else res

def __calc_cstr(A, b, c, w):
    """For w ∈ [0,1]ⁿ, returns the value of wᵀAw + bᵀw + c."""
    return __calc_obj(A, b, c, w)


def qubo_resolution(objs, cstrs, m):
    """
    Construct matrices for objectives and constraints, using given parameters.

    Parameters
    ----------
    objs : list [(type, [A, b, c]), ...]
        list of quadratic objectives, see examples below.
    cstrs : list [(type, [b, c]), ...]
        list of linear constraints, see examples below.
    m : int
        number of resolution bits.
    
    Returns
    -------
    mat_objs : list of numpy matrices
        list of matrices coding the objective functions.
    mat_cstrs : list of numpy matrices
        list of matrices coding the constraints.

    Examples
    --------
    *Problem 1:*  we want 6 resolution bits for solving
        min  wᵀAw + bᵀw + c
                          w ∈ [0,1]ⁿ
    `qubo_resolution([('min',[A,b,c])], [], m=6)`

    *Problem 2:*  we want 8 resolution bits for solving
        max             µᵀw
        min            wᵀΣw
        s.t.        𝟙ᵀw - 1 = 0
                          w ∈ [0,1]ⁿ
    `qubo_resolution([('max',[0,µ,0]), ('min',[Σ,0,0])], [('eq',[np.ones(N),-1])], m=8)`

    *Problem 3:*  we want to find w ∈ [0,1]ⁿ with 4 resolution bits, such that w is subject to
        s.t.  bᵀw = c
    `qubo_resolution([], [('eq',[b,-c])], m=4)`
    """

    mat_objs, mat_cstrs, v = __qubo_resolution_helper(objs, cstrs, m)
    return mat_objs, mat_cstrs

def __qubo_resolution_helper(objs, cstrs, m):
    """
    Helper method for :meth:`qubo_resolution()`, which additionally returns the scaling vector `v`.
    """

    # Helper vectors and matrices for binary scaling
    v = (1 << np.arange(m))[::-1] / ((1<<m)-1)   # v = (2ᵇ⁻¹, ..., 2, 1)ᵀ / (2ᵇ - 1)
    vvᵀ = np.outer(v, v)

    mat_objs = [__qubo_obj(A, b, c, v, vvᵀ) * (-1 if type.casefold()=='max' else 1) for (type, [A, b, c]) in objs]
    mat_cstrs = [__qubo_cstr(0, b, c, v, vvᵀ) * (1 if type.casefold()=='eq' else None) for (type, [b, c]) in cstrs] # TODO 
    return mat_objs, mat_cstrs, v


def __recombine_w(w, objs, cstrs):
    """Helper method for retrieving the objective and constraint values,
    given a sample vector `w`."""
    return (w,
            [__calc_obj(A, b, c, w) for (type, [A, b, c]) in objs],
            [__calc_cstr(0, b, c, w) for (type, [b, c]) in cstrs])

def __recombine_bin_vect(bin_vect, objs, cstrs, v):
    """Helper method for retrieving the objective and constraint values,
    given a sample binary vector `bin_vect` and its corresponding scaling vector `v`."""
    m = len(v)            # v = (2ᵇ⁻¹, ..., 2, 1)ᵀ / (2ᵇ - 1)
    bin_matrix = bin_vect.reshape(len(bin_vect)//m, m)
    w = bin_matrix @ v    # Retrieve the asset weights in percentage from the binary solution vector
    return __recombine_w(w, objs, cstrs)


def transform_into_bqm(matrix):
    """
    Turns a QUBO matrix into its corresponding :obj:`dimod.BinaryQuadraticModel`, which is used by the D-Wave packages.
    
    Parameters
    ----------
    matrix : a numpy matrix.
    
    Returns
    -------
    The corresponding :obj:`dimod.BinaryQuadraticModel` object.
    """

    N = len(matrix)
    qubo = {}
    for i in range(N):
        for j in range(N):
            if abs(matrix[i,j]) > 1e-8:
                qubo[(i,j)] = matrix[i,j]
    if (N-1, N-1) not in qubo:
        qubo[(N-1, N-1)] = 0

    return dimod.BQM.from_qubo(qubo)


def anneal(bqm, annealingSamples, annealingTime, DWtoken=None, DWregion="eu-central-1", steepest_descent=False):
    """
    Use D-Wave's hardware / simulated annealing algorithm to solve a given QUBO problem.
    Return the best sample as a binary vector.

    Parameters
    ----------
    bqm : :obj:`dimod.BinaryQuadraticModel`
        The input QUBO as a bqm object.
    annealingSamples : int
        Number of annealing samples (i.e. number of annealing runs / tests).
    annealingTime : float
        Time of annealing for each sample (only works for real quantum hardware).
    DWtoken : None / str, optional
        If given a string: use it as the token for accessing D-Wave quantum hardware;
        If None: use D-Wave's simulated annealing algorithm.
    DWregion : str, optional
        Choose to use D-Wave's compute resources in a particular region. Default is "eu-central-1".
    steepest_descent : bool, optional
        Whether to do a classical gradient descent optimization afterwards on the obtained samples.
    
    Returns
    -------
    bin_vect : numpy vector (datatype: uint8)
        The best sample as a binary vector.
    solving_time : float
        The time of annealing (excludes in-queue waiting time in case of real life hardware).
        For the time of quantum annealing on real D-Wave hardware, the solving time
        is calculated as 'qpu_access_time' + 'post_processing_overhead_time'.
    """

    if DWtoken is not None:
        sampler = EmbeddingComposite(DWaveSampler(token=DWtoken, region=DWregion))
        solution = sampler.sample(bqm, num_reads=annealingSamples, annealing_time=annealingTime)
        timing_info = solution.info['timing']
        solving_time = timing_info['qpu_access_time'] + timing_info['post_processing_overhead_time']
    else:
        sampler = neal.SimulatedAnnealingSampler()
        t1 = time.process_time_ns()
        solution = sampler.sample(bqm, num_reads=annealingSamples, annealing_time=annealingTime)
        t2 = time.process_time_ns()
        solving_time = (t2 - t1)/1000

    if steepest_descent:
        t3 = time.process_time_ns()
        solver = greedy.SteepestDescentSolver()
        solution = solver.sample(bqm, initial_states=solution)
        t4 = time.process_time_ns()
        solving_time += (t4 - t3)/1000

    bestSample = solution.first.sample
    bin_vect = np.fromiter(bestSample.values(), dtype='uint8')

    return bin_vect, solving_time


def solve_qubo(objs, cstrs, λ, P, m, annealingSamples=1, annealingTime=20, n_tests=1, DWtoken=None, DWregion="eu-central-1", steepest_descent=False, disp=False):
    """
    Use D-Wave's hardware / simulated annealing algorithm to solve a given QUBO problem.

    Parameters
    ----------
    objs : list [(type, [A, b, c]), ...]
        List of quadratic objectives, see examples below.
    cstrs : list [(type, [b, c]), ...]
        List of linear constraints, see examples below.
    λ : list of floats
        Scalarization factors / linearization factors for the objectives.
    P : list of floats
        Penalty factors for the constraints.
    m : int
        number of resolution bits.
    annealingSamples : int
        Number of annealing samples (i.e. number of annealing runs / tests).
    annealingTime : float
        Time of annealing for each sample (only works for real quantum hardware).
    n_tests : int
        Number of tests to be done.
        Difference from `annealingSamples`: a high number of annealingSamples
        only gives a higher
    DWtoken : None / str, optional
        If given a string: use it as the token for accessing D-Wave quantum hardware;
        If None: use D-Wave's simulated annealing algorithm.
    DWregion : str, optional
        Choose to use D-Wave's compute resources in a particular region. Default is "eu-central-1".

    Examples for `objs` and `cstrs`
    -------------------------------
    *Problem 1:*  we want 6 resolution bits for solving
        min  wᵀAw + bᵀw + c
                          w ∈ [0,1]ⁿ
    `qubo_resolution([('min',[A,b,c])], [], m=6)`

    *Problem 2:*  we want 8 resolution bits for solving
        max             µᵀw
        min            wᵀΣw
        s.t.        𝟙ᵀw - 1 = 0
                          w ∈ [0,1]ⁿ
    `qubo_resolution([('max',[0,µ,0]), ('min',[Σ,0,0])], [('eq',[np.ones(N),-1])], m=8)`

    *Problem 3:*  we want to find w ∈ [0,1]ⁿ with 4 resolution bits, such that w is subject to
        s.t.  bᵀw = c
    `qubo_resolution([], [('eq',[b,-c])], m=4)`

    Returns
    -------
    df : pandas dataframe
        A dataframe of solutions, where each row represents a single test.
        The columns are: the best solution from each test ('w'), objective function values
        ('obj_0', 'obj_1', ...), constraint function values ('cstr_0', 'cstr_1', ...),
        the respective solving time ('time') in microseconds (1 µs = 1e-6 s),
        and the scalarization factors used ('λ').
        For the time of quantum annealing on real D-Wave hardware, the solving time
        is calculated as 'qpu_access_time' + 'post_processing_overhead_time'.
    """

    a, b = len(objs), len(cstrs)
    assert(len(objs) == len(λ) and len(cstrs) == len(P))
    mat_objs, mat_cstrs, v = __qubo_resolution_helper(objs, cstrs, m)
    qubo_matrix = sum( λ[k]*mat_objs[k] for k in range(a) ) + sum( P[l]*mat_cstrs[l] for l in range(b) )
    bqm = transform_into_bqm(qubo_matrix)

    points = []
    for i in tqdm(range(n_tests), disable=(not disp or n_tests == 1)):
        bin_vect, solving_time = anneal(bqm, annealingSamples, annealingTime, DWtoken, DWregion, steepest_descent)
        w, val_objs, val_cstrs = __recombine_bin_vect(bin_vect, objs, cstrs, v)
        points.append([w] + val_objs + val_cstrs + [solving_time] + [λ])  # '+' is list concatenation
    
    if disp:
        if n_tests == 1:
            bin_matrix = bin_vect.reshape(len(bin_vect)//m, m)
            print(bin_matrix)
    df = __points_to_df(points, a, b, disp)
    return df

def __points_to_df(points, a, b, disp):
    """
    Helper method for formatting a list of points (with their respective objective and
    constraint values) into a dataframe.

    Parameters
    ----------
    points : list of vectors
        Sample points to evaluate. (Format of each point:
        ['w', 'obj_0', 'obj_1', ..., 'cstr_0', 'cstr_1', ..., 'time'])
    a : int
        Number of objective functions.
    b : int
        Number of constraints.
    disp : bool
        Whether to display the resulting dataframe in readable precision.
    
    Returns
    -------
    df : pandas dataframe
        The resulting dataframe.
    """

    df = pd.DataFrame(points, columns=['w'] + ['obj_'+str(k) for k in range(a)] + ['cstr_'+str(l) for l in range(b)] + ['time'] + ['λ'])
    if disp:
        with pd.option_context('display.precision', 3):
            df_disp = df.copy()
            df_disp['w'] = df_disp['w'].apply(lambda x: np.round(x, 3))
            display(df_disp)
    return df


#------------------------------
# Basic Classical Optimization
#------------------------------

def __get_N(objs, cstrs):
    """Helper method for getting the required vector length "n", given objectives and constraints."""
    for (type, [A, b, c]) in objs:
        if not (np.isscalar(A) and A == 0):
            return np.shape(A)[0]
        if not (np.isscalar(b) and b == 0):
            return np.shape(b)[0]
    for (type, [b, c]) in cstrs:
        if not (np.isscalar(b) and b == 0):
            return np.shape(b)[0]
    return None

def __rand_vec_default(N):
    """Default method for generating a random vector of length N."""
    w = np.random.rand(N)
    return w/np.sum(w)

def __qubo_obj_sco(A, b, c, w):
    """Returns a python function for w ↦ wᵀAw + bᵀw + c."""
    res = c
    if not (np.isscalar(A) and A == 0):
        res += w.T @ A @ w
    if not (np.isscalar(b) and b == 0):
        res += np.dot(b, w)
    return res

def __qubo_objs_sco(objs):
    """Returns a list of python functions calculating the given objectives."""
    return [ lambda w, A=A, b=b, c=c, type=type:  # A solution to deal with Python's late binding closures
                 __qubo_obj_sco(A, b, c, w) * (-1 if type.casefold()=='max' else 1)
             for (type, [A, b, c]) in objs ]

def __qubo_cstrs_sco(cstrs):
    """Converts a list of constraints in a respective format for `scipy.optimize`."""
    return ( {'type': type, 'fun': lambda w, b=b, c=c, type=type: __qubo_obj_sco(0, b, c, w)}
             for (type, [b, c]) in cstrs )

def solve_qubo_sco(objs, cstrs, λ, rand_vec=None, method=None, n_tests=1, disp=False):

    assert(len(λ) == len(objs))
    fun_objs = __qubo_objs_sco(objs)
    if rand_vec is None:
        rand_vec = __rand_vec_default
    
    N = __get_N(objs, cstrs)

    points = []
    for i in tqdm(range(n_tests), disable=(not disp or n_tests == 1)):
        t1 = time.process_time_ns()
        w0 = rand_vec(N)
        w0 = w0 / np.sum(w0)
        res = sco.minimize(
                fun=lambda w: sum(λ[k]*fun_objs[k](w) for k in range(len(fun_objs))),
                x0=w0, constraints=__qubo_cstrs_sco(cstrs), bounds=((0,1),)*N, method=method
            )
        w = res.x
        w, val_objs, val_cstrs = __recombine_w(w, objs, cstrs)
        t2 = time.process_time_ns()
        solving_time = (t2 - t1)/1000
        points.append([w] + val_objs + val_cstrs + [solving_time] + [λ])  # '+' is list concatenation
    
    df = __points_to_df(points, len(objs), len(cstrs), disp)
    return df


# ---------------------------
#  Multivariate Optimization
# ---------------------------

def __weighed_sum_helper(n, k):
    """List all possibilities of putting n indistinguishable objects into k bins."""
    if k == 0:
        return []
    return [np.array(list(map(operator.sub, cuts + (n,), (0,) + cuts))) for cuts in itertools.combinations_with_replacement(range(n+1), k-1)]

def efficient_frontier(objs, cstrs, method, **kwargs):
    """
    Compute the efficient frontier of a given multicriteria optimization problem.

    Parameters
    ----------
    objs : list [(type, [A, b, c]), ...]
        List of quadratic objectives, see examples below.
    cstrs : list [(type, [b, c]), ...]
        List of linear constraints, see examples below.
    method : string
        The following methods are currently supported:
        - 'weighed_sum':
            Weighed sum scalarization with `scipy.optimize`.
        - 'weighed_sum_annealing':
            Weighed sum scalarization with D-Wave's annealing hardware / software.
        - 'monte_carlo':
            Monte Carlo simulation.
    
    Returns
    -------
    mat_objs : list of numpy matrices
        list of matrices coding the objective functions.
    mat_cstrs : list of numpy matrices
        list of matrices coding the constraints.

    Examples for `objs` and `cstrs`
    -------------------------------
    *Problem 1:*  we want 6 resolution bits for solving
        min  wᵀAw + bᵀw + c
                          w ∈ [0,1]ⁿ
    `qubo_resolution([('min',[A,b,c])], [], m=6)`

    *Problem 2:*  we want 8 resolution bits for solving
        max             µᵀw
        min            wᵀΣw
        s.t.        𝟙ᵀw - 1 = 0
                          w ∈ [0,1]ⁿ
    `qubo_resolution([('max',[0,µ,0]), ('min',[Σ,0,0])], [('eq',[np.ones(N),-1])], m=8)`

    *Problem 3:*  we want to find w ∈ [0,1]ⁿ with 4 resolution bits, such that w is subject to
        s.t.  bᵀw = c
    `qubo_resolution([], [('eq',[b,-c])], m=4)`

    Returns
    -------
    df : pandas dataframe
        A dataframe of the resulting samlpes, where each row represents a single sample.
        The columns are: the best solution from each test ('w'), objective function values
        ('obj_0', 'obj_1', ...), constraint function values ('cstr_0', 'cstr_1', ...),
        the respective solving time ('time') in microseconds (1 µs = 1e-6 s),
        and the scalarization factors used ('λ').
        For the time of quantum annealing on real D-Wave hardware, the solving time
        is calculated as 'qpu_access_time' + 'post_processing_overhead_time'.
    """

    num_objs = len(objs)
    rand_vec = kwargs.get('rand_vec', __rand_vec_default)

    meth = method.casefold()
    if meth in ['weighed_sum', 'weighed sum', 'weighed_sum_annealing', 'weighed sum annealing']:

        n_scale = kwargs.get('n_scale', 50)
        k = kwargs.get('k', np.ones(num_objs)) / n_scale                   # pointwise division
        λs = [k * arr for arr in __weighed_sum_helper(n_scale, num_objs)]  # pointwise multiplication
        if len(λs) == 0:
            λs = [[]]

        dfs = [None] * len(λs)
        for i in tqdm(range(len(λs))):
            if meth in ['weighed_sum_annealing', 'weighed sum annealing']:
                dfs[i] = solve_qubo(objs, cstrs, λs[i], kwargs.get('P', [20]), kwargs.get('m', 6),
                                    kwargs.get('annealingSamples', 1), kwargs.get('annealingTime', 20), n_tests=kwargs.get('n_tests', 1),
                                    DWtoken=kwargs.get('DWtoken', None), DWregion=kwargs.get('DWregion', "eu-central-1"), steepest_descent=kwargs.get('steepest_descent', False))
            else:
                dfs[i] = solve_qubo_sco(objs, cstrs, λs[i], rand_vec=rand_vec, method=kwargs.get('method', None), n_tests=kwargs.get('n_tests', 1))
        df = pd.concat(dfs)

    elif meth in ['monte-carlo', 'mc', 'monte carlo', 'monte_carlo']:

        N = __get_N(objs, cstrs)
        num = kwargs.get('num', 10000)
        points = [None] * num                  # Initialize empty list
        for i in tqdm(range(num)):
            w = rand_vec(N)
            w, val_objs, val_cstrs = __recombine_w(w, objs, cstrs)
            points[i] = [w] + val_objs + val_cstrs + [np.nan] + [np.nan]  # '+' is list concatenation
        df = __points_to_df(points, num_objs, len(cstrs), disp=False)

    elif meth in ['eps_constraint' | 'constrained' | 'epsilon_constraint' | 'eps constraint' | 'epsilon constraint']:
        # TODO #############################################################################
        return

    else:
        print("Optimization method " + method + " not available!")
        return

    return df

    



#--------------------------
# Plots and visualizations
#--------------------------

def plot_return_volatility_matrices(μ, σ, cor, Σ, savedir=None):
    """
    TODO: Documentation
    """

    N = len(μ)

    plt.figure(figsize=(0.65*(2.5+2*N),0.65*N))
    gs = matplotlib.gridspec.GridSpec(1, 15+12*N)
    gs.update(wspace=0, hspace=0)
    ax0 = plt.subplot(gs[0:1, 0:6])
    ax1 = plt.subplot(gs[0:1, 7:13])
    ax2 = plt.subplot(gs[0:1, 14:(14+6*N)])
    ax3 = plt.subplot(gs[0:1, (15+6*N):(15+12*N)])
    annot_size = 9
    def plot_heatmap_helper(matrix, axis, title, xticklabels=False, yticklabels=False, cmap=cc.cm.CET_D9, annot=True, fmt='.0%'):
        hmap = sns.heatmap(
            matrix, ax=axis,
            cmap=cmap, center=0, cbar=False,
            annot=annot, annot_kws={'size': annot_size}, fmt=fmt,
            xticklabels=xticklabels, yticklabels=yticklabels, # robust=True,
            linecolor='lightgray', linewidth=.3, rasterized=False, square=False
        )
        axis.set_title(title, fontsize=12, pad=7.5)
        axis.set_xticklabels(hmap.get_xticklabels(), rotation=0, fontsize=annot_size)
        axis.set_yticklabels(hmap.get_yticklabels(), rotation=0, fontsize=annot_size)
        axis.tick_params(axis='both', which='both', length=1)
        axis.axhline(y=0, color='k',linewidth=1)
        axis.axhline(y=matrix.shape[0], color='k', linewidth=1.5)
        axis.axvline(x=0, color='k',linewidth=1)
        axis.axvline(x=matrix.shape[1], color='k', linewidth=1.5)
        return hmap
    vwg = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#AB5EDD","#F8F7F8","#53942E"])
    #greyscale = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FAFAFA","#FAFAFA","#666666"])
    plot_heatmap_helper(μ[:,None], ax0, "μ", fmt='.1%', cmap=vwg, yticklabels=True)
    plot_heatmap_helper(σ[:,None], ax1, "σ", fmt='.1%', cmap=vwg)
    ax0.set_yticklabels(list(range(1,N+1)))
    plot_heatmap_helper(cor, ax2, "Correlation Matrix", xticklabels=True)
    ax2.set_xticklabels(list(range(1,N+1)))
    plot_heatmap_helper(Σ, ax3, "Covariance Matrix", cmap=vwg, xticklabels=True, yticklabels=True)
    ax3.set_xticklabels(list(range(1,N+1)))
    ax3.set_yticklabels(list(range(1,N+1)))
    ax3.yaxis.tick_right()

    if savedir is not None:
        plt.savefig(savedir, dpi=300, bbox_inches='tight')
    plt.show()


def plot_matrices(dict, figsize, cbar=True, annot=False, fmt='.2g', ticklabels=False, linewidths=.0, subtitlesize=12):
    """
    Plot matrix heatmaps, using a dictionary (axes of plot -> (matrix to plot, its subtitle)) as input.
    
    Parameters
    ----------
    dict : dictionary ((int, int) -> (numpy matrix, str))
        Dictionary mapping the axes of each heatmap plot to the corresponding matrix and its subtitle.
    figsize : (float, float)
        Width, height of the whole plot, in inches.
    cbar : bool, optional
        Whether to draw a colorbar next to each heatmap.
    annot : bool, optional
        Whether to write the data value in each cell.
    fmt : str, optional
        String formatting code for the data values.
    ticklabels : bool, optional
        Whether to show the row and column indices.
    linewidths : float, optional
        Width of the lines that will divide each cell.
    subtitlesize : float, optional
        Font size of the subtitles above each heatmap. 
    """

    fig, axs = plt.subplots(
        ncols=max([k[1] for k in dict.keys()])+1,
        nrows=max([k[0] for k in dict.keys()])+1,
        squeeze=False, figsize=figsize
        )
    
    for k,v in dict.items():
        axis, matrix, title = axs[k], v[0], v[1]
        sns.heatmap(
            matrix, ax=axis,
            cmap=cc.cm.CET_D9, center=0, cbar=cbar, cbar_kws={'aspect': 30, 'pad': 0.04, 'fraction': 0.04},
            annot=annot, fmt=fmt,
            xticklabels=ticklabels, yticklabels=ticklabels,
            linewidths=linewidths, linecolor='#F8F7F8', square=True
        ).set_title(title, fontsize=subtitlesize, pad=7.5)
        if cbar:
            axis.collections[0].colorbar.ax.tick_params(labelsize=9)
        axis.axhline(y=0, color='k',linewidth=1)
        axis.axhline(y=matrix.shape[0], color='k', linewidth=1.5)
        axis.axvline(x=0, color='k',linewidth=1)
        axis.axvline(x=matrix.shape[1], color='k', linewidth=1.5)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0)


def frontiers_plotter(dflist, titlelist, df_background=None, color='g', s_foreground=10, s_background=3, save_as=None):
    obj_names = [col for col in dflist[0] if col.startswith('obj')]
    num_objs = len(obj_names)
    combs = list(itertools.combinations(range(num_objs), 2))
    num_rows = len(combs) # = num_objs * (num_objs-1) // 2

    fig = plt.figure(figsize=(3*len(dflist),3*num_rows), layout='constrained')
    subfigs = fig.subfigures(ncols=len(dflist), nrows=1, squeeze=False)
    if not (df_background is None or s_background == 0):
        lims = [1.05*max(max(max(dflist[i][n]) for i in range(len(dflist))), df_background[n]) for n in obj_names]
    else:
        lims = [1.05*max(max(dflist[i][n]) for i in range(len(dflist))) for n in obj_names]
    for i in range(len(dflist)):
        subfigs[0][i].suptitle(titlelist[i])
        axs = subfigs[0][i].subplots(ncols=1, nrows=num_rows, squeeze=False)
        for j in range(num_rows):
            ax = axs[j][0]
            df_foreground = dflist[i]
            x_name = obj_names[combs[j][1]]
            y_name = obj_names[combs[j][0]]
            viol_list = df_foreground.get('cstr_0')
            color_list = ['r' if b else color for b in viol_list] if viol_list is not None else [color]
            sns.scatterplot(x=df_foreground[x_name], y=df_foreground[y_name], ax=ax, c=color_list, linewidth=0, s=s_foreground)
            if not (df_background is None or s_background == 0):
                sns.scatterplot(x=df_background[x_name], y=df_background[y_name], ax=ax, color='lightgray', linewidth=0, s=s_background)
            ax.set_xlim(0, lims[combs[j][1]])
            ax.set_ylim(0, lims[combs[j][0]])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    fig.set_constrained_layout_pads(h_pad=8/72, w_pad=4/72)
    if not save_as is None:
        #plt.savefig(os.path.join("images", save_as), dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(os.path.join("", save_as), dpi=300, bbox_inches='tight', transparent=True)
    fig.set_constrained_layout_pads(h_pad=8/72, w_pad=4/72)
    plt.show()


#--------
# Others
#--------

def generate_correlation_matrix(d):
    """
    Generate a symmetric, positive definite correlation matrix.
    Source: https://stats.stackexchange.com/q/125017

    Parameters
    ----------
    d : int
        The side length of the correlation matrix.
    
    Returns
    -------
    S : numpy matrix
        The generated correlation matrix.
    """
    S = np.eye(1)
    for k in range(2, d+1):
        y = np.random.beta(a=(k-1)/2, b=(d+1-k)/2)     # sampling from beta distribution
        r = math.sqrt(y)
        theta = np.random.randn(k-1, 1)
        theta = theta/np.linalg.norm(theta)
        w = r*theta
        E,U = np.linalg.eig(S)
        R = U @ np.diag(E**(1/2)) @ np.transpose(U)    # R is a square root of S
        q = R @ w
        S = np.vstack((
                np.hstack((S, q)),
                np.hstack((np.transpose(q), [[1]]))))  # increasing the matrix size
    return S