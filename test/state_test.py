from common import subsets
import fastfermion
from fastfermion import MajoranaString, paulis, paulistrings, fermistrings, FockState, QubitProductState
import numpy as np

def test_fockstate():
    assert FockState().vec(1) == [1,0]
    assert FockState([0]).vec(1) == [0,1]
    assert FockState([]).vec(2) == [1,0,0,0]
    assert FockState([1]).vec(2) == [0,1,0,0]
    assert FockState([0]).vec(2) == [0,0,1,0]
    assert FockState([0,1]).vec(2) == [0,0,0,1]
    m = 4
    for fs in fermistrings(m):
        M = fs.sparse(m)
        for B in subsets(range(m)):
            state = FockState(B)
            state_vec = state.vec(m)
            val1 = state(fs)
            val2 = np.dot(state_vec, M.dot(state_vec))
            assert abs(val1 - val2) <= 1e-14
            if fs.degree(0) == fs.degree(1):
                # Particle-number preserving
                state_vec_sub = state.vec(m,len(B))
                Msub = fs.sparse(m,len(B))
                val3 = np.dot(state_vec_sub, Msub.dot(state_vec_sub))
                assert abs(val3 - val2) <= 1e-14
            

    for S in subsets(range(2*m)):
        ms = MajoranaString(S)
        M = ms.sparse(m)
        for B in subsets(range(m)):
            state = FockState(B)
            state_vec = state.vec(m)
            val1 = state(ms)
            val2 = np.dot(state_vec, M.dot(state_vec))
            assert abs(val1 - val2) <= 1e-14

def test_qubitproductstate():
    assert QubitProductState().vec(1) == [1,0]
    assert QubitProductState([0]).vec(1) == [0,1]
    assert QubitProductState([]).vec(2) == [1,0,0,0]
    assert QubitProductState([1]).vec(2) == [0,1,0,0]
    assert QubitProductState([0]).vec(2) == [0,0,1,0]
    assert QubitProductState([0,1]).vec(2) == [0,0,0,1]

    n = 4
    for ps in paulistrings(n):
        M = ps.sparse(n)
        for B in subsets(range(n)):
            state = QubitProductState(B)
            state_vec = state.vec(n)
            val1 = state(ps)
            val2 = np.dot(state_vec, M.dot(state_vec))
            assert abs(val1 - val2) <= 1e-14

    # Construct polynomial which preserves total spin
    sx, sy, sz = paulis(10)
    def h(i,j):
        return sx[i]*sx[j] + sy[i]*sy[j] + sz[i]*sz[j]
    # Some polynomial which preserves total spins
    p = sum(np.sin(1+i)*np.cos(2+j)*h(i,j) for i in range(n) for j in range(n))
    Mp = [p.sparse(n,nup) for nup in range(n+1)]
    for B in subsets(range(n)):
        state = QubitProductState(B)
        state_vec = state.vec(n,len(B))
        val1 = state(p)
        val2 = np.dot(state_vec, Mp[len(B)].dot(state_vec))
        assert abs(val1 - val2) <= 1e-14
