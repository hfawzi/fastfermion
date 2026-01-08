"""
Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
All rights reserved. Use of this source code is governed
by a license that can be found in the LICENSE file.

fastfermion/Cirq interface
"""

import cirq
from . import ffcore as ff

def from_cirq(circuit: cirq.Circuit):
    """Converts a Cirq circuit into a fastfermion list of gates"""
    res = []
    for moment in circuit:
        for op in moment.operations:
            qubits = [int(q) for q in op.qubits]
            gate = op.gate
            # Cirq's Gate zoo: https://quantumai.google/cirq/gatezoo
            if gate == cirq.H:
                res.append(ff.H(qubits[0]))
            elif gate == cirq.S:
                res.append(ff.S(qubits[0]))
            elif gate == cirq.CNOT:
                res.append(ff.CNOT(qubits[0],qubits[1]))
            elif gate == cirq.CZ:
                res.append(ff.CZ(qubits[0],qubits[1]))
            elif gate == cirq.SWAP:
                res.append(ff.SWAP(qubits[0],qubits[1]))
            elif isinstance(gate,cirq.Rx):
                res.append(ff.ROT("X",[qubits[0]],gate._rads))
            elif isinstance(gate,cirq.Ry):
                res.append(ff.ROT("Y",[qubits[0]],gate._rads))
            elif isinstance(gate,cirq.Rz):
                res.append(ff.ROT("Z",[qubits[0]],gate._rads))
            else:
                raise ValueError(f"Cirq gate {gate} not supported by fastfermion")
    return res

def to_cirq(circuit: list):
    """Converts a fastfermion list of gates to a Cirq circuit"""
    qubits_range = cirq.LineQubit.range(ff.MAX_QUBITS)
    res = cirq.Circuit()
    for gate in circuit:
        qubits = [qubits_range[q] for q in gate.qubits]
        if isinstance(gate, ff.H):
            res.append(cirq.H(qubits[0]))
        elif isinstance(gate, ff.S):
            res.append(cirq.S(qubits[0]))
        elif isinstance(gate, ff.CNOT):
            res.append(cirq.CNOT(qubits[0],qubits[1]))
        elif isinstance(gate, ff.CZ):
            res.append(cirq.CZ(qubits[0],qubits[1]))
        elif isinstance(gate, ff.SWAP):
            res.append(cirq.SWAP(qubits[0],qubits[1]))
        elif isinstance(gate, ff.ROT):
            if len(gate.qubits) == 1:
                if gate.axis.degree("X") == 1:
                    res.append(cirq.Rx(rads=gate.theta)(qubits[0]))
                elif gate.axis.degree("Y") == 1:
                    res.append(cirq.Ry(rads=gate.theta)(qubits[0]))
                elif gate.axis.degree("Z") == 1:
                    res.append(cirq.Rz(rads=gate.theta)(qubits[0]))
    return res

def to_paulisum(poly: ff.PauliString | ff.PauliPolynomial):
    """Converts a fastfermion PauliPolynomial to a Cirq PauliSum"""
    res = cirq.PauliSum()
    qubits = cirq.LineQubit.range(ff.MAX_QUBITS)
    def to_cirq_paulistring(ps: ff.PauliString, coeff = 1):
        return cirq.PauliString(
            dict(
                (qubits[ind], op) for ind, op in ps.indices()
            ), coeff)
    if isinstance(poly, ff.PauliString):
        return to_cirq_paulistring(poly)
    for ps,coeff in poly.terms.items():
        res += to_cirq_paulistring(ps,coeff)
    return res
