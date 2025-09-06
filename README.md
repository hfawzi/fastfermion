# FastFermion

**fastfermion** is a Python package written in C++ for the efficient manipulation of polynomials in Pauli, Fermi and Majorana operators.

<p align="center">
<img alt="Computing the Jordan-Wigner transform of a CrO molecule Hamiltonian with > 10^5 terms" src="jwperf.svg" style="max-height: 70px;" /><br />
<i>
Computing the Jordan-Wigner transform of a CrO molecule Hamiltonian with > 10<sup>5</sup> terms</i>
</p>

**Features**

* Algebraic manipulation of polynomials in Pauli operators, Fermionic creation/annihilation operators, and Majorana operators.
* Fermionic and Majorana operators are automatically put in normal ordered form
* Conversion between Pauli, Fermi, and Majorana representations (Jordan-Wigner and reverse Jordan-Wigner)
* Sparse matrix representations
* Heisenberg evolution: Propagate polynomial through a sequence of unitaries/gates with possible truncation by degree
* Interface with OpenFermion and Cirq
* Up to 100x faster than OpenFermion
* More to come ...

**Important**: The package is still under development and so the API may change at any time.

## Installation

fastfermion is available on PyPI:

```shell
pip3 install fastfermion
```

## Getting started

* [First steps](#first-steps)
* [Transforms](#transforms)
* [Sparse representations](#sparse-matrix-representations)
* [Unitaries and evolution](#unitaries-and-evolution)
* [Interface with Cirq and OpenFermion](#interface-with-openfermion-and-cirq)

### First steps

Let's start by constructing a Pauli polynomial.

```python
>>> from fastfermion import poly
>>> A = poly("X0 X1 + (.3 X0 Z1 + .4 Y0 Y1) Z2")
```

The above expression creates a Pauli polynomial with three terms. One can get the number of terms in $A$ using the `len` function, and individual monomial coefficients using the `coefficient` function.

```python
>>> len(A)
3
>>> A.coefficient("X0 Z1 Z2")
(0.3+0j)
```

Polynomials can be added and multiplied using the usual operations. Several properties of a polynomial can be accessed via functions such as `degree`, `norm`, `coefficient`, etc.

```python
>>> A += 2
>>> print(A)
X0 X1  +  0.3 X0 Z1 Z2  +  0.4 Y0 Y1 Z2  +  2
>>> A.norm('inf')
2.0
>>> A.degree()
3
>>> A.degree("Y")
2
```

To construct $A$ we used the function `poly` which parses a string into a fastfermion polynomial. Alternatively, one can use `paulis` to get the generators of the Pauli algebra and use standard Python operations to build up a polynomial. For example, to construct the Heisenberg Hamiltonian on a line, one can proceed as follows.

```python
>>> from fastfermion import paulis
>>> L = 4
>>> X,Y,Z = paulis(L)
>>> edges = [(i,i+1) for i in range(L-1)]
>>> H = sum(X[i]*X[j] + Y[i]*Y[j] + Z[i]*Z[j] for i,j in edges)
>>> print(H)
X0 X1  +  Y0 Y1  +  Z0 Z1  +  X1 X2  +  Y1 Y2  +  Z1 Z2  +  X2 X3  +  Y2 Y3  +  Z2 Z3
```

Importantly, fastfermion also supports fermionic polynomials in creation/annihilation operators. Fermionic creation/annihilation operators $f_p^{\dagger}$, $f_p$ obey the canonical anticommutation relations:

```math
\{f_p,f_q\} = 0, \quad \{f_p^{\dagger}, f_q\} = \delta_{pq}.
```

We can also use the `poly` function to input the polynomial as a string.

```python
>>> from fastfermion import poly
>>> B = poly("f0 f0^ + 3.1 f2^ f0 + 1")
>>> print(B)
- f0^ f0  +  2  +  3.1 f2^ f0
>>> B.coefficient("f2^ f0")
(3.1+0j)
```

The notation `f0^` indicates a creation operator and `f0` (without a trailing `^`) indicates an annihilation operator.
Note that the expression $f_0f_0^{\dagger}$ was automatically changed to $1-f_0^{\dagger} f_0$ in the internal representation of $B$. This is because fastfermion stores all fermionic polynomials in normal ordered form: all the creation operators appear to the left of annihilation operators, and creation and annihilation operators are ordered in *decreasing order* from left to right.

Fermionic polynomials can also be constructed by first getting generators of the algebra (annihilation operators) using the `fermis` command, and then building up a polynomial. The code below creates a quadratic polynomial $\sum_{ij} f_i^{\dagger} f_j$:

```python
>>> from fastfermion import fermis
>>> L = 5
>>> edges = [(i,i+1) for i in range(L-1)]
>>> f = fermis(L)
>>> H = sum(f[i].dagger() * f[j] for i,j in edges)
>>> print(H)
f0^ f1  +  f1^ f2  +  f2^ f3  +  f3^ f4
```

Finally, fastfermion also supports polynomials in Majorana operators $m_p$, which are Hermitian operators that obey the following anticommutation relation:

```math
\{ m_p,m_q \} = 2 \delta_{pq}.
```

One can construct Majorana polynomials in the same way we constructed Pauli polynomials or Fermi polynomials.

```python
>>> from fastfermion import poly, majoranas
>>> C = poly("(m0 m1 m4 + 0.2 m2 m1) m4")
>>> print(C)
m0 m1  -  0.2 m1 m2 m4
>>> m = majoranas(5)
>>> D = (m[0]*m[1]*m[4] + 0.2*m[2]*m[1])*m[4]
>>> C-D
0
```

Polynomials in Majorana operators are also automatically put in normal ordered form. In this case, normal form means that Majorana modes appear in *increasing order* from left to right. This is why the second term in $C$, which was input $m_2 m_1 m_4$ is represented as $-m_1 m_2 m_4$ instead.

### Transforms

fastfermion offers the possibility to transform polynomials from one type {Pauli, Fermi, Majorana} to any other type.
The well-known Jordan-Wigner transform transforms polynomials in Fermionic creation/annihilation operators to a polynomial in Pauli operators using the rule:

$$
\begin{aligned}
f_j &\leftarrow Z_0 \dots Z_{j-1} (X_j + \text{i} Y_j) / 2\\
f_j^{\dagger} &\leftarrow Z_0 \dots Z_{j-1} (X_j - \text{i} Y_j) / 2.
\end{aligned}
$$

The Jordan-Wigner transform can be computed via the function `jw`. Furthermore the reverse Jordan-Wigner transform can be computed using the function `rjw`.

```python
>>> from fastfermion import poly, jw, rjw
>>> B = poly("f0 f0^ + 3.1 f2^ f0 + 1")
>>> jw(B)
1.5  +  0.5 Z0  +  0.775j Y0 Z1 X2  +  0.775 X0 Z1 X2  +  0.775 Y0 Z1 Y2  -  0.775j X0 Z1 Y2
>>> rjw(jw(B)) - B
0
```

One can also convert any Fermi polynomial to a Majorana polynomial via the following transform

$$
\begin{aligned}
f_j &\leftarrow (m_{2j} + \text{i} m_{2j+1})/2\\
f_j^{\dagger} &\leftarrow (m_{2j} - \text{i} m_{2j+1}) / 2.
\end{aligned}
$$

```python
>>> B.tomajorana()
1.5  -  0.5j m0 m1  -  0.775 m0 m4  -  0.775j m1 m4  +  0.775j m0 m5  -  0.775 m1 m5
```

The functions `topauli()`, `tofermi()`, `tomajorana()` can be used to go from one type of polynomial to another.


### Sparse matrix representations

Given any polynomial in fastfermion, one can use the `sparse` function to get a sparse matrix representation of the corresponding polynomial.

```python
>>> from fastfermion import poly
>>> A = poly("X0*X1 + Z0 + 3")
>>> A.sparse()
<Compressed Sparse Column sparse matrix of dtype 'complex128'
        with 8 stored elements and shape (4, 4)>
>>> A.sparse().toarray()
array([[4.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 4.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 2.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 2.+0.j]])
```

One can check that the above matrix is indeed equal to $\sigma_x \otimes \sigma_x + \sigma_z \otimes I_2 + 3 I_2 \otimes I_2$, where $\sigma_x$, $\sigma_y$ and $\sigma_z$ are the standard $2\times 2$ Pauli matrices.


The `sparse()` function can also be used for Fermi polynomials and Majorana polynomials.

```python
>>> from fastfermion import poly
>>> poly("f1^ f1 f0^ - 2 f1 f1^ + 3").sparse()
<Compressed Sparse Column sparse matrix of dtype 'complex128'
        with 5 stored elements and shape (4, 4)>
>>> poly("m0 m1 - m2 m0").sparse()
<Compressed Sparse Column sparse matrix of dtype 'complex128'
        with 8 stored elements and shape (4, 4)>
```

Note that a Fermi polynomial with $n$ modes is represented (in the occupation number basis) as a $2^n \times 2^n$ matrix, and a Majorana polynomial on $n$ variables is represented as a $2^{\lceil n/2 \rceil} \times 2^{\lceil n/2 \rceil}$ matrix.

### Unitaries and evolution

A unitary is any operator $U$ such that $U^{\dagger} U = 1$. Such a unitary acts on polynomials by conjugation: $P \to U^{\dagger} P U$.
fastfermion currently supports a limited number of built-in unitaries for Pauli polynomials, namely `CNOT`, `H`, `S`, and `ROT`. For example:


```python
>>> from fastfermion import poly, CNOT
>>> A = poly("X0 X1 + (.3 X0 Z1 + .4 Y0 Y1) Z2")
>>> CNOT(0,1)(A) # Apply the CNOT(0,1) unitary to A
X0  -  0.3 Y0 Y1 Z2  -  0.4 X0 Z1 Z2
```

A circuit is an ordered collection of unitaries $C = (U_1,\ldots,U_d)$ where $d$ is the depth of the circuit. Starting from a polynomial $P$, the evolution of $P$ through the circuit is given by

$$
U_1^{\dagger} \dots U_d^{\dagger} P U_d \dots U_1.
$$

The `propagate` function is a direct method to propagate a Pauli polynomial through a circuit.

```python
>>> from fastfermion import poly, CNOT, H, S, ROT, propagate
>>> P = poly('X0')
>>> circuit = [CNOT(0,1), H(1), ROT("ZZ",[0,1],0.785)]
>>> Q = propagate(circuit, P)
>>> Q
0.7073882691671998 X0 X1 - 0.706825181105366 Y0
```

The `propagate` function is equivalent to the following simple for loop:

```python
def propagate(circuit, poly):
    for gate in reversed(circuit):
        poly = gate(poly)
    return poly
```

For large circuits, one can specify an additional parameter to `propagate` that truncates the degree of the polynomial after the action of non-Clifford gates (namely `ROT` gates).

```python
>>> S = propagate(circuit, P, maxdegree=3)
```

### Interface with OpenFermion and Cirq

The functions `from_openfermion` and `to_openfermion` can be used to convert from/to OpenFermion operators.
Also the function `from_cirq` can be used to convert a Cirq circuit to a list of FastFermion gates.

```python
>>> import cirq
>>> from fastfermion import from_cirq
>>> q0, q1 = cirq.LineQubit.range(2)
>>> cirq_circuit = cirq.Circuit(cirq.H(q0),cirq.CNOT(q0, q1), cirq.Rx(rads=0.23)(q0))
>>> ff_circuit = from_cirq(cirq_circuit)
>>> ff_circuit
[H(0), CNOT(0,1), ROT(X0,0.230000)]
```

Currently, only the Cirq gates `cirq.H`, `cirq.S`, `cirq.CNOT`, `cirq.Rx`, `cirq.Ry`, `cirq.Ry` are supported.

## Reference

Below is a summary of the list of functions and methods currently supported by fastfermion. For more details about how to use these functions, use `help(<function name>)` in the Python console.

### Constructing polynomials

| Function | |
| --- | --- |
| `poly` | Parse string expression
| `paulis` | Get 3N Pauli generators |
| `fermis` | Get annihilation operators |
| `majoranas` | Get Majorana operators |
| `from_openfermion` | Convert from OpenFermion |
| `to_openfermion` | Convert to OpenFermion |


### Operations on polynomials

| Function | |
| --- | --- |
| `degree` | Degree of polynomial |
| `dagger` | Adjoint of a polynomial |
| `coefficient` | Get monomial's coefficient |
| `norm` | Norm of the coefficient vector |
| `truncate` | Remove high degree terms |
| `compress` | Remove terms with small weight |
| `extent` | Number of variables in a polynomial |
| `permute` | Permute variables |
| `sparse` | Sparse matrix representation of polynomial |
| `commutes` | Check if two polynomials commute |
| `commutator` | Commutator of two polynomials |

> **_NOTE:_** Most of the functions above (except `commutes` and `commutator`) can be invoked using the dot notation. For example, if $A$ is a {Pauli,Fermi,Majorana} polynomial, then `degree(A)` or `A.degree()` are equivalent.

### Transforms

| Function | |
| --- | --- |
| `topauli` | Convert to Pauli polynomial |
| `tofermi` | Convert to Fermi polynomial |
| `tomajorana` | Convert to Majorana polynomial |
| `jw` | Jordan-Wigner transform (Fermi -> Pauli) |
| `rjw` | Reverse Jordan-Wigner transform (Pauli -> Fermi) |

> **_NOTE:_**  If $A$ is a Fermi polynomial then `jw(A)` and `A.topauli()` are equivalent. Similarly if $A$ is a Pauli polynomial then `rjw(A)` and `A.tofermi()` are equivalent.

> **_NOTE:_**  The functions `topauli`, `tofermi`, `tomajorana` can also be invoked with the dot notation. For example, if $A$ is a Fermi polynomial, then `A.tomajorana()` and `tomajorana(A)` are equivalent.

### Unitaries (Gates) and evolution

In fastfermion, a circuit is simply represented as a list of unitaries. Currently we only support the following four types of unitaries `H`, `S`, `CNOT`, `ROT`.

| Function | |
| --- | --- |
| `H` | Hadamard gate |
| `S` | S gate |
| `CNOT` | CNOT gate |
| `ROT` | Pauli rotation $U=e^{-i \theta/2 P}$ where $P$ is a Pauli string |
| `from_cirq` | Convert Cirq circuit into a list of fastfermion unitaries |
| `propagate` | Propagate a Pauli polynomial through a circuit (Heisenberg evolution) |

