import fastfermion as ff

def read_circuit(filename: str) -> list:
    """
    Reads a Pauli circuit from a file
    The first line of the file is the number of gates
    Each subsequent line represents a gate. Example:
    4
    CNOT 0 1
    ROT X 1 0.444
    ROT XY 1 3 0.32
    H 0
    Returns
        [ff.CNOT(0,1), ff.ROT("X",[1],0.444), ...]
    """
    circuit = []
    with open(filename,'r',encoding='utf-8') as f:
        # Read number of gates
        _ = int(f.readline())
        for line in f:
            gate_info = line.split(' ')
            gate_type = gate_info[0] # Either CNOT, ROT, H, S
            params = gate_info[1:]
            # Cast params to correct type to instantiate gate
            if gate_type == "ROT":
                # Special case for ROT since the constructor takes
                # the qubits as a list
                gate = ff.ROT(params[0],
                              [int(q) for q in params[1:-1]],
                              float(params[-1].rstrip()))
            else:
                # params is a list of int
                gate = getattr(ff, gate_type)(*[int(q) for q in params])
            circuit.append(gate)
    return circuit

def read_fermipolynomial(filename: str) -> ff.FermiPolynomial:
    """
    Reads a FermiPolynomial from a file
    Each term of the polynomial appears on a line
    Example:
    4^ 2^ 1 0 0.54
    1 -0.18
    0.56
    3^ 1.12
    represents the polynomial
    0.56 I + 0.54 f4^ f2^ f1 f0 - 0.18 f1 + 1.12 f3^
    """
    q = ff.FermiPolynomial()
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"):
                continue
            term = line.strip().split(' ')
            coefficient = complex(term[-1])
            term_ops = tuple(
                (int(op[:-1]),1) if op[-1] == '^' else (int(op),0)
                for op in term[:-1])
            q += coefficient * ff.FermiString(term_ops)
    return q
