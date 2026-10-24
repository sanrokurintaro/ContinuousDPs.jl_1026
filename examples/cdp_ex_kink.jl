# A two-dimensional model whose value function has a C^1 kink at a known
# location, approximated with `kinked_breaks` (a spline breakpoint of
# multiplicity 2 at the kink).
#
# State s = (a, logy): assets and log income (AR(1)); action x = a', next
# period's assets. A soft borrowing penalty
#
#     psi(a) = -(kappa / 2) * min(a, 0)^2
#
# enters the reward. It is C^1 at a = 0 (psi'' jumps from -kappa to 0), so
# the value function is C^1 with a jump in V'' on the line a = 0: an
# EXOGENOUS kink on an axis of the state space.
#
# A cubic spline is C^2 at an ordinary breakpoint, so a single breakpoint at
# a = 0 cannot represent the kink. `kinked_breaks` (from the package) repeats
# the breakpoint, which lowers the continuity there to C^1 and matches the
# value function. Chebyshev polynomials and an evenly spaced spline both smear
# the kink. This script compares them against a fine-grid reference.
using ContinuousDPs
using BasisMatrices: Basis, ChebParams, SplineParams
using QuantEcon: qnwnorm

a_min, a_max = -2.0, 6.0
logy_min, logy_max = -0.5, 0.5
gamma, beta, rho, sigma, R, kappa = 2.0, 0.92, 0.9, 0.1, 1.03, 5.0

u(c) = c^(1 - gamma) / (1 - gamma)
psi(a) = -0.5 * kappa * min(a, 0.0)^2

f(s, x) = u(R * s[1] + exp(s[2]) - x) + psi(s[1])
# Keep the next state inside the approximation domain
g(s, x, e) = (clamp(x, a_min, a_max), clamp(rho * s[2] + e, logy_min, logy_max))
x_lb(s) = a_min
x_ub(s) = min(a_max, R * s[1] + exp(s[2]) - 1e-3)
shocks, weights = qnwnorm(5, 0.0, sigma^2)

cdp = ContinuousDP(f, g, beta, shocks, weights, x_lb, x_ub)

# The basis of the kinked dimension (a) is the only thing that changes
# between the candidates below; logy always uses a Chebyshev basis.
logy_basis = ChebParams(7, logy_min, logy_max)
solve_with(a_params) =
    solve(cdp, CollocationSolver(Basis(a_params, logy_basis); algorithm=VFI,
                                 max_iter=2000); verbose=0)

# Reference solution on a fine grid, compared on a test grid
test_points = [(a, ly) for ly in (-0.3, 0.0, 0.3)
               for a in range(-1.9, 5.0, length=691)]
function evaluate(res)
    pf = ContinuousDPs.PolicyFunction(res)
    vf = ContinuousDPs.ValueFunction(res)
    return [pf(p) for p in test_points], [vf(p) for p in test_points]
end

x_ref, v_ref = evaluate(solve_with(
    SplineParams(kinked_breaks(a_min, a_max, [0.0]; n=161), 0, 3)))

println("max abs error against the reference (a-direction basis size n)")
for n in (24, 48)
    candidates = (
        "Chebyshev" => ChebParams(n, a_min, a_max),
        "spline, even breaks" =>
            SplineParams(kinked_breaks(a_min, a_max, (); n=n), 0, 3),
        "spline, knot at 0 (x1)" =>
            SplineParams(kinked_breaks(a_min, a_max, [0.0]; n=n, mult=1), 0, 3),
        "spline, knot at 0 (x2)" =>
            SplineParams(kinked_breaks(a_min, a_max, [0.0]; n=n, mult=2), 0, 3),
    )
    for (label, params) in candidates
        x, v = evaluate(solve_with(params))
        println("n = ", n, "  ", rpad(label, 24),
                "value ", round(maximum(abs.(v .- v_ref)), sigdigits=2),
                "  policy ", round(maximum(abs.(x .- x_ref)), sigdigits=2))
    end
end
