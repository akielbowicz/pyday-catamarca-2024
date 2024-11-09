from juliacall import Main as jl

s = r"""
using LinearAlgebra

function solve(A, b)
    A\b
end
"""

jl.seval(s)
solve_julia = jl.solve
