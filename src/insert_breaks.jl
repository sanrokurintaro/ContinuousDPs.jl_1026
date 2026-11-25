#=
Inserting breakpoints into spline and piecewise linear bases, e.g. at the
locations of known kinks of the value function.

A spline of degree `k` is `C^(k-1)` at an ordinary (simple) breakpoint, so it
cannot represent a kink there. Repeating a breakpoint `m` times lowers the
continuity at that point to `C^(k-m)`: a cubic spline (`k = 3`) is `C^1` at a
double breakpoint and `C^0` at a triple one. Placing repeated breakpoints at
the kinks therefore lets a spline basis resolve them, while the smooth parts
keep the usual spline accuracy. A piecewise linear basis is `C^0` at every
breakpoint, so inserting a simple breakpoint at the kink is enough there.

The functions take and return `BasisMatrices` parameter objects; no other part
of the package needs to know about them.
=#

using BasisMatrices: BasisParams, SplineParams, LinParams, Basis

# Sorted `Float64` locations, checked to be distinct and strictly inside
# `(lb, ub)`
function _check_locs(locs, lb, ub)
    ls = sort!(Float64[l for l in locs])
    all(l -> lb < l < ub, ls) || throw(ArgumentError(
        "locs must lie strictly inside ($lb, $ub)"))
    allunique(ls) || throw(ArgumentError("locs must be distinct"))
    return ls
end

"""
    insert_breaks(p::SplineParams, locs; multiplicity=1) -> SplineParams
    insert_breaks(p::LinParams, locs) -> LinParams
    insert_breaks(basis::Basis, dim::Integer, locs; multiplicity=1) -> Basis

Return a copy of the basis parameters `p` (or of `basis`, with the parameters
of dimension `dim` replaced) with breakpoints added at the points in `locs`.

A value function that is `C^1` with a jump in the second derivative at a known
point (for example, at the threshold of a smooth borrowing penalty) is poorly
approximated by Chebyshev polynomials or by a spline with no breakpoint there,
because both smear the kink. Repeating the breakpoint at the kink lowers the
continuity of the spline there so that the kink can be represented.

# Arguments

- `p`: `SplineParams` or `LinParams`. The result has explicit breakpoints
  (`evennum = 0`), also when `p` was evenly spaced; the degree of a spline is
  kept.
- `basis::Basis`, `dim::Integer`: Basis and the dimension whose parameters are
  modified; the other dimensions are unchanged. Chebyshev dimensions have no
  breakpoints and are rejected.
- `locs`: Collection of locations, distinct and strictly inside the interval
  covered by `p`; pass `()` for none (`p` is returned unchanged).
- `multiplicity::Integer`: Multiplicity each location has in the result. For a
  spline of degree `k`, `multiplicity = 1` gives `C^(k-1)` (no kink),
  `multiplicity = 2` gives `C^(k-2)`, and so on down to `multiplicity = k` for
  `C^0`; it must not exceed `k`. A location that is already a breakpoint of
  `p` is repeated up to `multiplicity` in total, never reduced. A piecewise
  linear basis is `C^0` at every breakpoint, so it takes no `multiplicity`
  (through a `Basis`, only `multiplicity = 1` is accepted for such a
  dimension); locations that are already breakpoints are left as they are.

# Examples

```julia
using BasisMatrices: Basis, SplineParams, ChebParams

p = SplineParams(range(-2.0, 6.0, length=24), 0, 3)
p = insert_breaks(p, [0.0]; multiplicity=2)         # C^1 at 0 for a cubic

# several dimensions: modify the kinked one
basis = Basis(SplineParams(range(-2.0, 6.0, length=24), 0, 3),
              ChebParams(7, -0.5, 0.5))
basis = insert_breaks(basis, 1, [0.0]; multiplicity=2)
```
"""
function insert_breaks(p::SplineParams, locs; multiplicity::Integer=1)
    1 <= multiplicity <= p.k || throw(ArgumentError(
        "multiplicity must be between 1 and the spline degree k = $(p.k)"))
    b = collect(Float64, p.breaks)
    ls = _check_locs(locs, b[1], b[end])
    isempty(ls) && return p
    for l in ls
        append!(b, fill(l, max(multiplicity - count(==(l), b), 0)))
    end
    return SplineParams(sort!(b), 0, p.k)
end

function insert_breaks(p::LinParams, locs)
    b = collect(Float64, p.breaks)
    ls = _check_locs(locs, b[1], b[end])
    isempty(ls) && return p
    return LinParams(sort!(unique!(vcat(b, ls))), 0)
end

function insert_breaks(basis::Basis{N}, dim::Integer, locs;
                       multiplicity::Integer=1) where {N}
    1 <= dim <= N || throw(ArgumentError(
        "dim must be between 1 and the number of dimensions $N"))
    new_params = ntuple(N) do i
        i == dim ? _insert_breaks(basis.params[i], locs, multiplicity) :
                   basis.params[i]
    end
    return Basis(new_params...)
end

_insert_breaks(p::SplineParams, locs, multiplicity) =
    insert_breaks(p, locs; multiplicity=multiplicity)

function _insert_breaks(p::LinParams, locs, multiplicity)
    multiplicity == 1 || throw(ArgumentError(
        "a piecewise linear basis takes multiplicity = 1"))
    return insert_breaks(p, locs)
end

_insert_breaks(p::BasisParams, locs, multiplicity) = throw(ArgumentError(
    "insert_breaks supports SplineParams and LinParams, not $(typeof(p))"))
