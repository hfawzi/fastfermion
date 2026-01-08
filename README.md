[![PyPi](https://img.shields.io/pypi/v/fastfermion.svg)](https://pypi.python.org/pypi/fastfermion/)


**fastfermion** is a Python package written in C++ for the efficient manipulation of polynomials in Pauli, Fermi and Majorana operators.

<p align="center">
<img alt="Computing the Jordan-Wigner transform of a CrO molecule Hamiltonian with > 10^5 terms" src="assets/jwperf.svg" style="height: 120px;" /><br />
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
* Up to 200x faster than OpenFermion
* More to come ...

## Installation

fastfermion is available on PyPI:

```shell
pip3 install fastfermion
```

### Building from source

Assuming you have a modern C++ compiler, simply run from the root directory of the package:

```shell
make ffcore
```

This will create a binary file `ffcore...` inside the `fastfermion` subdirectory.
To import the package in Python, just add the root fastfermion directory in your path, e.g.,

```python
>>> import sys
>>> sys.path.insert(0,"/path/to/fastfermion")
>>> import fastfermion
```

You could also use the library directly in your C++ project (even though the library was primarily intended to be used in Python). It is header-only, so you can just include the relevant header files from `src/`.

## Resources

* [A tour of fastfermion](https://www.fastfermion.com/tour/)
* Examples: see the `examples/` folder
