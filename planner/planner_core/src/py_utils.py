import numpy as np
from scipy.stats import mvn, entropy
import random
from numpy import linalg as la


def correct_zeros(arr):
    for i in range(len(arr)):
        if arr[i] < 1e-7:
            arr[i] = 1e-7
    return arr

## Discrete PDF Distances
def sym_KL(p, q):
    try:
        p = correct_zeros(p)
        q = correct_zeros(q)
        return entropy(p, q) + entropy(q, p)
    except ValueError as valerr:
        print valerr
        raise


def KL(p, q):
    try: 
        p = correct_zeros(p)
        q = correct_zeros(q)
        return entropy(p, q)
    except ValueError as valerr:
        print valerr
        raise


def Bhattacharya(p, q):
    try:
        p /= np.sum(p)
        q /= np.sum(q)
        res = np.zeros(len(p))
        for i in range(len(p)):
            res[i] = np.sqrt(p[i]*q[i])
        return -np.log(np.sum(res))
    except ZeroDivisionError:
        print "Zero Division Error in probablities. Check Inputs"
        raise
    except ValueError as valerr:
        print valerr
        raise

def Hellinger(p,q):
    try:
        p /= np.sum(p)
        q /= np.sum(q)
        res = np.zeros(len(p))
        for i in range(len(p)):
            res[i] = (np.sqrt(p[i]) - np.sqrt(q[i]))**2
        return np.sqrt(2*np.sum(res))
    except ZeroDivisionError:
        print "Zero Division Error in probablities. Check Inputs"
        raise
    except ValueError as valerr:
        print valerr
        raise

def smooth_cost(p,q, cost_type='Hellinger', scale=1e6):
    d_t = 0.0
    d_b = 2.0

    if cost_type == 'KL':
        cost = KL(p,q)
    elif cost_type == 'sym_KL':
        cost = sym_KL(p,q)
    elif cost_type == 'Bhattacharya':
        cost = Bhattacharya(p,q)
    else:
        cost = Hellinger(p,q)

    if cost < d_t:
        return 0.
    elif (cost >=d_t and cost<=d_b):
        return scale*(d_b*((cost-d_t)**2/(d_b-d_t)**2))
    else:
        return scale*cost



### Utilities
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    Code Ref: https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # print "spacing =", spacing
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky 
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        k += 1
        A3 += I * (-mineig * k**2 + spacing)

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True and np.all(np.linalg.eigvals(B) > 0)
    except la.LinAlgError:
        return False
    # return np.all(np.linalg.eigvals(B) > 0)

# def PDSQ(A):
#     print "Input Mat = ", A
#     mat = A.T.dot(A)
#     print "Output Mat = ", la.cholesky(mat)
#     return la.cholesky(mat)


def simpleCdf(nState, low, high, mu, cov):
	neg_inf = -1e4*np.ones(nState)
	res = abs( mvn.mvnun(neg_inf, high, mu, cov)[0] - mvn.mvnun(neg_inf, low, mu, cov)[0])
	return res


def multilinspace(x1, x2, num=10, endpoint=True):
    pts = np.zeros((len(x1), num))
    newpts = []
    for i in range(len(x1)):
        pts[i,:] = np.linspace(x1[i], x2[i], num=num, endpoint=endpoint)
    for j in range(num):
        newpts.append(np.array([pts[i,j] for i in range(len(x1))]))
    return newpts


def initialGuess(mu, goal, nPts=10):
    mid_point = np.array([-2, (mu[1]+goal[1])/2, (mu[2]+goal[1])/2, (mu[3]+goal[1])/2])
    if (nPts%2 == 0):
        nPts_new1 = nPts/2
        nPts_new2 = nPts_new1*1
    else:
        nPts_new1 = (nPts+1)/2
        nPts_new2 = (nPts_new1 - 1)*1

    pt1 = multilinspace(mu, mid_point, num=nPts_new1)
    pt2 = multilinspace(mid_point, goal, num=nPts_new2)    
    new_pts = np.append(pt1, pt2)
    return new_pts


def warmStart(filename, nPts):
    data = np.load(filename)
    b_traj = data['belief_traj']
    step = len(b_traj)/nPts
    if step < 1.:
        mat1 = b_traj.flatten()
        mat2 = np.array([])
        for i in range(nPts-len(b_traj)):
            mat2 = np.append(mat2,np.random.rand(np.shape(b_traj)[1]) + b_traj[-1,:])
        mat2 = np.array(mat2)
        return np.append(mat1, mat2)
    else:
        muInit = np.array([b_traj[np.int(np.round(i))] for i in np.arange(0., len(b_traj), step)])
        muInit = muInit[:nPts,:]
        return muInit.flatten('F')


# Generating symmetric array from upper trangular elements
class SymNDArray(np.ndarray):
    def __setitem__(self, (i, j), value):
        super(SymNDArray, self).__setitem__((i, j), value)                    
        super(SymNDArray, self).__setitem__((j, i), value)                    

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def symarray(input_array):
    """
    Returns a symmetrized version of the array-like input_array.
    Further assignments to the array are automatically symmetrized.
    """
    return symmetrize(np.asarray(input_array)).view(SymNDArray)


## Adaptive step size based onthe information
def adaptiveSegments(x, domain, tFinal, min_seg):
    adapN = min_seg + tFinal*(np.linalg.norm(x)/np.linalg.norm(domain))   
    # adapN = abs(tFinal - tFinal*(cov_0/9.5))
    return int(np.ceil(adapN))


def change_contrast(array, level):
    array1 = array*1.
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)   
    for i in range(np.shape(array)[1]):
        for j in range(np.shape(array)[1]):
            array1[i][j] = contrast(array[i][j])
    
    # remapping to 0-256
    array1 = array1/(array1.max()-array1.min())
    return (array1 + abs(array1.min()))*500


class L(list):
    def __init__(self, sz=3):
        self.sz = sz

    def append(self, item):
        list.append(self, item)
        if len(self) > self.sz: self[:1]=[]

def set_covariance_matrix(matrix, indices, value):
    mat = np.zeros(len(matrix))
    for idx in indices:
        mat[idx] = value
    print "Updated Matrix: ", mat
    return mat
