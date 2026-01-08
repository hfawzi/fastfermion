from common import check_paulipolynomial_approx_equal
import itertools
import cirq
import numpy as np
import fastfermion as ff
from fastfermion import CNOT, ROT, H, S, SWAP, propagate

def clifford_gate_test_helper(gate: str, num_qubits: int):
    """Test correctness of Clifford gate implementation
    against Cirq. Generates a list of PauliStrings, runs them on the gate
    and checks the output matches Cirq
    num_qubits is the number of qubits the gate acts on
    num_qubits = 1 if gate = H or S, and num_qubits = 2 if gate = CNOT or SWAP
    """

    # Extract the gate class from ff dynamically
    n = 3
    ff_gate_function = getattr(ff, gate)
    cirq_gate_function = getattr(cirq, gate)
    cirq_qubits = cirq.LineQubit.range(n)

    def check_paulimonomial_cirq_equal(
        a: tuple[ff.PauliString,complex],
        b: cirq.PauliString
    ):
        """Checks that a ff PauliMonomial and a cirq.PauliString
        are equal"""
        assert a[0] == ff.PauliString(str(b.dense(cirq_qubits)).strip("+-i").replace("_","I")) \
            and a[1] == b.coefficient, f"The two paulistrings are not equal {a} != {b}"
    
    
    all_strings = ["".join(s) for s in itertools.product(("I","X","Y","Z"),repeat=n)]

    for qubits in itertools.product(range(n),repeat=num_qubits):
        if len(set(qubits)) == num_qubits:
            # Ensure the qubits we are acting on are distinct
            ff_gate = ff_gate_function(*qubits)
            cirq_op = cirq_gate_function(*[cirq_qubits[q] for q in qubits])
            for s in all_strings:
                ff_out = ff_gate(ff.PauliString(s))
                cirq_out = cirq.DensePauliString(s).on(*cirq_qubits).before(cirq_op)
                # print(f"{gate} on {s} gives ff: {ff_out}, cirq: {cirq_out}")
                check_paulimonomial_cirq_equal(ff_out, cirq_out)

def test_clifford_gates():
    """Check Clifford gates implementation"""
    clifford_gate_test_helper("H",1)
    clifford_gate_test_helper("S",1)
    clifford_gate_test_helper("CNOT",2)
    clifford_gate_test_helper("SWAP",2)
    clifford_gate_test_helper("CZ",2)

def simple_propagate_1(circuit: list, obs: ff.PauliPolynomial):
    """Simple propagation algorithm used to test the propagate function"""
    for gate in reversed(circuit):
        gate_poly = gate.aspoly()
        obs = gate_poly.dagger() * obs * gate_poly
        obs = obs.compress(0)
    return obs

def simple_propagate_2(circuit: list, obs: ff.PauliPolynomial):
    """Simple propagation algorithm used to test the propagate function"""
    for gate in reversed(circuit):
        obs = gate(obs)
        obs = obs.compress(0)
    return obs

def test_propagate():
    """Test propagate function on small circuit, without truncation"""
    circuits = [
        [H(0)],
        [S(0)],
        [CNOT(0,1)],
        [H(0),CNOT(0,1),S(1),CNOT(1,2),H(2)],
        [ROT("I",0.3)],
        [ROT("XX",0.34)],
        [H(0),CNOT(0,1),ROT("XY",0.23),ROT("ZZ",0.78),S(1),CNOT(1,2),H(2)],
    ]
    observables = [
        ff.poly("X0 Z1"), ff.poly("X0Z1Y2"), ff.poly("-Z2+3*X0")
    ]
    for circuit in circuits:
        for observable in observables:
            simple_propagation_res_1 = simple_propagate_1(circuit, observable)
            simple_propagation_res_2 = simple_propagate_2(circuit, observable)
            propagate_res = propagate(circuit, observable)
            check_paulipolynomial_approx_equal(
                simple_propagation_res_1, propagate_res
            )
            check_paulipolynomial_approx_equal(
                simple_propagation_res_2, propagate_res
            )

            # Now compare with Cirq
            # q0, q1 = cirq.LineQubit.range(2)
            num_qubits = max(
                max(max(q for q in gate.qubits) if len(gate.qubits) > 0 else 0 for gate in circuit)+1,
                observable.extent()
            )
            print(circuit)
            print(observable)
            print(num_qubits)
            simulator = cirq.Simulator(dtype=np.complex128)
            ev = simulator.simulate_expectation_values(
                ff.to_cirq(circuit),
                observables=[ff.to_paulisum(observable)],
                qubit_order=cirq.LineQubit.range(num_qubits)
            )

            assert abs(ev[0] - propagate_res.overlapwithzero()) < 1e-6, \
                "Error computing expectation value"

def test_tocirq():
    q0, q1 = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(cirq.H(q0),cirq.CNOT(q0, q1), cirq.Rx(rads=0.23)(q0))
    assert ff.to_cirq(ff.from_cirq(cirq_circuit)) == cirq_circuit
