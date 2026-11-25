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
# [added] inserting (repeated) breakpoints, e.g. at kinks of the value function
include("insert_breaks.jl")

export
    ContinuousDP, solve, VFI, PFI, LQA,
    CollocationSolver, LQASolver,
    ActionSpace, ContinuousActions, DiscreteActions, ActionInterval,
    set_eval_nodes!, simulate, simulate!, approx_lq,
    insert_breaks  # [added]

end # module
