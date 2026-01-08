import platform
import os

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ffversionenv = os.getenv("FF_VERSION")
if ffversionenv is not None and ffversionenv.startswith("v"):
    __version__ = ffversionenv[1:]
else:
    __version__ = "0.1.0"

if platform.system().startswith('Win'):
    common_compile_args = [
        '/std:c++20',
        '/O2'
    ]
else:
    common_compile_args = [
        '-std=c++2a',
        '-O3'
    ]

ffcore = Pybind11Extension('fastfermion.ffcore',
    language='c++',
    extra_compile_args=[*common_compile_args] + [f'-DFF_VERSION="{__version__}"'],
    include_dirs=["src/","src/hashmap/"],
    sources=["src/python_bind.cpp"]
)

LONG_DESCRIPTION = """# FastFermion

**fastfermion** is a fast library written in C++ for manipulating polynomials in Pauli, Fermi and Majorana operators.

**Features**

* Algebraic manipulation of polynomials in Pauli operators, Fermionic creation/annihilation operators, and Majorana operators.
* Fermionic and Majorana operators are automatically put in normal ordered form
* Conversion between Pauli, Fermi, and Majorana representations (Jordan-Wigner and reverse Jordan-Wigner)
* Sparse matrix representations
* Heisenberg evolution: Propagate polynomial through a sequence of unitaries/gates with possible truncation by degree
* Interface with OpenFermion and Cirq
* Up to 100x faster than OpenFermion
* More to come ...

See https://github.com/hfawzi/fastfermion for more details
"""

# https://setuptools.pypa.io/en/latest/deprecated/distutils/setupscript.html
setup(
    name='fastfermion',
    version=__version__,
    author='Hamza Fawzi',
    author_email='hamzafawzi@gmail.com',
    url='https://github.com/hfawzi/fastfermion',
    description='A library for manipulating polynomials in Pauli, Fermi and Majorana operators.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    ext_modules=[ffcore],
    python_requires='>=3.10',
    packages=['fastfermion'],
    package_dir={'fastfermion': 'fastfermion'},
    install_requires=['scipy'],
)
