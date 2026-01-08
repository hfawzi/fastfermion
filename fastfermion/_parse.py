"""
Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
All rights reserved. Use of this source code is governed
by a license that can be found in the LICENSE file.

_parse.py: Implements fastfermion's poly function which parses
an expression and converts it to a fastfermion polynomial object
"""

import re
import ast

from . import ffcore as ff

fermi_symbol = ff.FERMI_SYMBOL
majorana_symbol = ff.MAJORANA_SYMBOL
dagger_symbol = ff.DAGGER_SYMBOL

def escape_re(symbol):
    if symbol == '^':
        return r'\^'
    return symbol

pauli_string_re = re.compile(r"((?:[XYZ][0-9]+[*]?)*(?:[XYZ][0-9]+))")
fermi_string_re = re.compile(
    rf"((?:{fermi_symbol}[0-9]+[{escape_re(dagger_symbol)}]?[*]?)*(?:{fermi_symbol}[0-9]+[{escape_re(dagger_symbol)}]?))"
)
majorana_string_re = re.compile(
    rf"((?:{majorana_symbol}[0-9]+[*]?)*(?:{majorana_symbol}[0-9]+))"
)


def poly(
    e: str,
) -> ff.PauliString | ff.PauliPolynomial | ff.FermiPolynomial | ff.MajoranaPolynomial:
    """
    Parse an expression and return corresponding fastfermion
    polynomial

    Example:
    >>> poly('X0 Z2')
    X0 Z2
    >>> poly('3.1 (X1 + Y1) Z1')
    -3.1j Y1 + 3.1j X1
    >>> poly('f3^ f1^ f0 f1')
    -1 f3^ f1^ f1 f0
    >>> poly('m1 m3 m2')
    -1 m1 m2 m3
    """

    # Fill in the missing '*' in the mathematical expression
    # Inspired from https://stackoverflow.com/questions/30010598/best-way-to-add-implied-multiplication-to-a-python-string
    # The three parts in the regex replace are "before (", "after )", and
    # "between digit or 'j' (complex numbers) or '.' (floating point)
    #  and one of (X,Y,Z,f,m)".
    # \w (word character) matches any single letter, number or underscore
    # (?<=a)b is a lookbehind that will match the b (and only the b) in cab
    # but does not match bed or debt
    enew = re.sub(
        rf"(?<=\w|\))(?=\() | (?<=\))(?=\w) | (?<=\d|j|\.)(?=[XYZ{fermi_symbol}{majorana_symbol}]) | (?<=\d{escape_re(dagger_symbol)})(?={fermi_symbol})",
        "*",
        e.replace(" ", ""),
        flags=re.X,
    )

    # Identify PauliStrings, FermiStrings and MonomialStrings and replace them
    # by a constructor call to the respective classes

    def pauli_replacer(match):
        group = match[0]
        indops = tuple((int(factor[1:]), factor[0]) for factor in group.split("*"))
        return f"PauliPolynomial({indops})"

    def fermi_replacer(match):
        group = match[0]
        indops = tuple(
            (int(factor[1:-1]), 1)
            if factor[-1] == dagger_symbol
            else (int(factor[1:]), 0)
            for factor in group.split("*")
        )
        return f"FermiPolynomial({indops})"

    def majorana_replacer(match):
        group = match[0]
        supp = tuple(int(factor[1:]) for factor in group.split("*"))
        return f"MajoranaPolynomial({supp})"

    enew = re.sub(pauli_string_re, pauli_replacer, enew)
    enew = re.sub(fermi_string_re, fermi_replacer, enew)
    enew = re.sub(majorana_string_re, majorana_replacer, enew)

    # Use python's ast to parse enew
    tree = ast.parse(enew)

    # Traverse tree to evaluate it
    def evalnode(node):
        if isinstance(node, ast.Module):
            if len(node.body) == 1:
                return evalnode(node.body[0])
            raise ValueError("Invalid expression")
        elif isinstance(node, ast.Expr):
            return evalnode(node.value)
        elif isinstance(node, ast.Call):
            assert len(node.args) == 1
            if node.func.id == "PauliPolynomial":
                return ff.PauliPolynomial(ast.literal_eval(node.args[0]))
            if node.func.id == "FermiPolynomial":
                return ff.FermiPolynomial(ast.literal_eval(node.args[0]))
            if node.func.id == "MajoranaPolynomial":
                return ff.MajoranaPolynomial(ast.literal_eval(node.args[0]))
            raise ValueError("Invalid expression")
        elif isinstance(node, ast.BinOp):
            left = evalnode(node.left)
            right = evalnode(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Div):
                return left / right
            raise ValueError("Invalid expression")
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                return evalnode(node.operand)
            if isinstance(node.op, ast.USub):
                return -evalnode(node.operand)
            raise ValueError("Invalid expression")
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            raise ValueError("Invalid expression")

    return evalnode(tree)
