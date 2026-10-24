#=
Spline breakpoints for value functions with kinks at known locations.

A spline of degree `k` is `C^(k-1)` at an ordinary (simple) breakpoint, so it
cannot represent a kink there. Repeating a breakpoint `mult` times lowers the
continuity at that point to `C^(k-mult)`: a cubic spline (`k = 3`) is `C^1` at a
double breakpoint and `C^0` at a triple one. Placing repeated breakpoints at
the kinks therefore lets a spline basis resolve them, while the smooth parts
keep the usual spline accuracy.

The result is meant for `BasisMatrices.SplineParams(breaks, 0, k)`; no other
part of the package needs to know about it.
=#

"""
    kinked_breaks(lb, ub, kinks; n, mult=2)

Return spline breakpoints on `[lb, ub]` with the points in `kinks` repeated
`mult` times each, for use with `SplineParams(breaks, 0, k)`.

A value function that is `C^1` with a jump in the second derivative at a known
point (for example, at the threshold of a smooth borrowing penalty) is poorly
approximated by Chebyshev polynomials or by a spline with evenly spaced
breakpoints, because both smear the kink. Repeating the breakpoint at the kink
lowers the continuity of the spline there so that the kink can be represented.

# Arguments

- `lb::Real`, `ub::Real`: Lower and upper bounds of the interval, `lb < ub`.
- `kinks`: Collection of kink locations, each strictly inside `(lb, ub)`;
  pass `()` for none (evenly spaced breakpoints).
- `n::Integer`: Number of distinct breakpoints, endpoints and kinks included.
  They are spread over the pieces between kinks in proportion to the piece
  lengths, with at least one subinterval per piece. The returned vector is
  longer than `n` by `(mult - 1) * length(kinks)`.
- `mult::Integer`: Multiplicity of each kink. For a spline of degree `k`,
  `mult = 1` gives `C^(k-1)` (no kink), `mult = 2` gives `C^(k-2)`, and so on
  down to `mult = k` for `C^0`. It must not exceed `k`; the default `2` suits a
  cubic spline and a value function with a jump in its second derivative.

# Returns

- `Vector{Float64}`: Sorted breakpoints.

# Examples

```julia
using BasisMatrices: Basis, SplineParams

breaks = kinked_breaks(-2.0, 6.0, [0.0]; n=24)          # 25 entries: 0.0 twice
basis = Basis(SplineParams(breaks, 0, 3))
```

For a state with several dimensions, give the kinked dimension this basis
and combine it with the others as usual, e.g.
`Basis(SplineParams(breaks, 0, 3), ChebParams(7, -0.5, 0.5))`.
"""
function kinked_breaks(lb::Real, ub::Real, kinks; n::Integer, mult::Integer=2)
    lb < ub || throw(ArgumentError("lb must be smaller than ub"))
    mult >= 1 || throw(ArgumentError("mult must be at least 1"))
    ks = sort!(Float64[k for k in kinks])
    all(k -> lb < k < ub, ks) || throw(ArgumentError(
        "kinks must lie strictly inside (lb, ub)"))
    allunique(ks) || throw(ArgumentError("kinks must be distinct"))

    # Pieces between consecutive kinks: [lb, k1], [k1, k2], ..., [kK, ub]
    edges = vcat(Float64(lb), ks, Float64(ub))
    n_pieces = length(edges) - 1
    n - 1 >= n_pieces || throw(ArgumentError(
        "n must be at least length(kinks) + 2 (= $(n_pieces + 1))"))

    # Subintervals per piece: one each, and the rest in proportion to length
    # (largest-remainder rounding so that the total is exactly n - 1)
    lens = diff(edges)
    extra = n - 1 - n_pieces
    share = extra .* lens ./ sum(lens)
    cnt = 1 .+ floor.(Int, share)
    leftover = (n - 1) - sum(cnt)
    order = sortperm(share .- floor.(share); rev=true)
    for i in order[1:leftover]
        cnt[i] += 1
    end

    breaks = Float64[edges[1]]
    for i in 1:n_pieces
        pts = range(edges[i], edges[i+1], length=cnt[i] + 1)
        append!(breaks, pts[2:end])
    end

    # Repeat each kink; the endpoints are handled by SplineParams itself
    mult == 1 && return breaks
    return sort!(vcat(breaks, [k for k in ks for _ in 1:(mult - 1)]))
end
