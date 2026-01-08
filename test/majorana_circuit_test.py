from common import check_majoranapolynomial_approx_equal
import fastfermion as ff
from fastfermion import MROT, MajoranaString, propagate
from scipy.linalg import expm, norm

def simple_propagate_1(circuit: list, obs: ff.MajoranaPolynomial):
    """Simple propagation algorithm used to test the propagate function"""
    for gate in reversed(circuit):
        gate_poly = gate.aspoly()
        obs = gate_poly.dagger() * obs * gate_poly
        obs = obs.compress(0)
    return obs

def simple_propagate_2(circuit: list, obs: ff.MajoranaPolynomial):
    """Simple propagation algorithm used to test the propagate function"""
    for gate in reversed(circuit):
        obs = gate(obs)
        obs = obs.compress(0)
    return obs

def test_propagate():
    """Test propagate function on small circuit, without truncation"""
    circuits = [
        [MROT(MajoranaString([0,2]),0.2j,0.1),MROT(MajoranaString([0,1,2,3]),0.3,0.1)],
    ]
    observables = [
        ff.poly("m0 m1"),
        ff.poly("-m1 + m0m2"),
        ff.poly("-1j*m2 + 2.1 m1 m0")
    ]
    for circuit in circuits:
        for observable in observables:
            simple_propagation_res_1 = simple_propagate_1(circuit, observable)
            simple_propagation_res_2 = simple_propagate_2(circuit, observable)
            propagate_res = propagate(circuit, observable)
            check_majoranapolynomial_approx_equal(
                simple_propagation_res_1, propagate_res
            )
            check_majoranapolynomial_approx_equal(
                simple_propagation_res_2, propagate_res
            )

def test_MROT_1():
    """
    Test constructor of MROT: MROT(MajoranaString,theta)
    """
    unitaries = [
        MROT(MajoranaString([]),0.3),
        MROT(MajoranaString([1]),0.3),
        MROT(MajoranaString([0,1]),0.3),
        MROT(MajoranaString([0,1,3]),0.2),
        MROT(MajoranaString([0,1,2,3]),0.2)
    ]
    A = ff.poly("0.1 m0 m1 + 1j m0 - 4 m1")
    for U in unitaries:
        Up = U.aspoly()
        check_majoranapolynomial_approx_equal(Up.dagger() * A * Up, U(A))
        # Check correctness on matrices
        Um = Up.sparse().toarray()
        r = (U.axis.degree() // 2) % 2 # 0 if U.axis Hermitian, 1 else
        Vm = expm(-1j * U.theta/2 * (1j)**r * U.axis.sparse().toarray())
        assert norm(Um-Vm) <= 1e-14, "Error matrices are not equal"

def test_MROT_2():
    """
    Test constructor of MROT: MROT(MajoranaString,coeff,theta)
    """
    ms = MajoranaString([0,1])
    coeff = 0.2j
    t = 0.3
    U = MROT(ms,coeff,t)
    Um = U.aspoly().sparse().toarray()
    Vm = expm(-1j * t/2 * coeff * ms.sparse().toarray())
    assert norm(Um-Vm) <= 1e-14, "Error matrices are not equal"
        