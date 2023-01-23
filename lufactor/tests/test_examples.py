import numpy as np
from ..lufactor import lu_crout

def test_lu_crout_2x2():
    A = np.array([[1.0, 7.0],
                  [2.0, 1.0]])

    Q = lu_crout(A)

    # decompose Q into LU
    L = Q - np.triu(Q, k=1)
    U = np.triu(Q, k=1) + np.identity(np.shape(Q)[0])

    # multiply LU and check if equal to A
    Aprime = np.matmul(L,U)
    isequal = np.all(np.isclose(Aprime, A))
    assert isequal

def test_lu_crout_2x2_pivot_needed():
    A = np.array([[1.0e-16, 1.0],
                  [    2.0, 1.0]])

    Q = lu_crout(A)

    # decompose Q into LU
    L = Q - np.triu(Q, k=1)
    U = np.triu(Q, k=1) + np.identity(np.shape(Q)[0])

    # multiply LU and check if equal to A
    Aprime = np.matmul(L,U)
    isequal = np.all(np.isclose(Aprime, A))
    assert isequal


def test_lu_crout_4x4():
    A = np.array([[ 6.0, -2.0,2.0,  4.0],
                  [12.0, -8.0,4.0, 10.0],
                  [ 3.0,-13.0,3.0,  3.0],
                  [-6.0,  4.0,2.0,-18.0]])

    Q = lu_crout(A)

    # decompose Q into LU
    L = Q - np.triu(Q, k=1)
    U = np.triu(Q, k=1) + np.identity(np.shape(Q)[0])

    # multiply LU and check if equal to A
    Aprime = np.matmul(L,U)
    isequal = np.all(np.isclose(Aprime, A))
    assert isequal

def test_lu_crout_1000x1000():
    np.random.seed(5)
    A = np.random.rand(1000,1000)

    Q = lu_crout(A)

    # decompose Q into LU
    L = Q - np.triu(Q, k=1)
    U = np.triu(Q, k=1) + np.identity(np.shape(Q)[0])

    # multiply LU and check if equal to A
    Aprime = np.matmul(L,U)
    isequal = np.all(np.isclose(Aprime, A))
    assert isequal