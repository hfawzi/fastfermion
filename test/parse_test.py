import common # To set the path for fastfermion
from fastfermion import FermiString, MajoranaString, PauliString, PauliPolynomial, poly


def test_poly():
    """Test poly parsing"""
    assert poly('X0 Z4 Y1') == PauliPolynomial(PauliString('XYIIZ'))
    assert poly('3.1 (X1 + Y1) Z1') == -3.1j * PauliString(((1,'Y'),)) + 3.1j * PauliString(((1,'X'),))
    assert poly('X0 X0') == 1 * PauliString()
    assert poly('f3^ f1^ f0 f1') == -FermiString([3,1],[1,0])
    assert poly('(3.12 * m10 m1 m2 - m3) m2') == -3.12*MajoranaString([1,10]) + MajoranaString([2,3])
    assert poly('1.5j X0 + 2.1 Z1') == 1.5j*PauliString("X") + 2.1 * PauliString("IZ")
    assert poly('1.5 j X0 + 2.1 Z1') == 1.5j*PauliString("X") + 2.1 * PauliString("IZ")
    assert poly('1.j X0 + 2. Z1') == 1.0j*PauliString("X") + 2.0 * PauliString("IZ")
    assert poly('-3 + f5^ f1 f3') == - 3 - FermiString([5],[3,1])
    assert poly('-3 + f15^ f21 f3') == - 3 + FermiString([15],[21,3])