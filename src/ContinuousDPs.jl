module ContinuousDPs

# stdlib
using LinearAlgebra, Random

using QuantEcon
import QuantEcon:
    bellman_operator!, DDPAlgorithm, VFI, PFI, solve,
    simulate, simulate!

const DPAlgorithm = DDPAlgorithm

include("point_eval.jl")
include("cdp.jl")
include("transition_kernel.jl")
include("inner_solvers.jl")
include("policy_system.jl")
include("lq_approx.jl")
# [added] spline breakpoints with repeated knots for kinked value functions
include("kinked_breaks.jl")

export
    ContinuousDP, solve, VFI, PFI, LQA,
    CollocationSolver, LQASolver,
    ActionSpace, ContinuousActions, DiscreteActions, ActionInterval,
    set_eval_nodes!, simulate, simulate!, approx_lq,
    kinked_breaks  # [added]

end # module
