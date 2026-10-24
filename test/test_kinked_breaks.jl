using BasisMatrices: BasisMatrix, Expanded, funeval

@testset "kinked_breaks" begin
    @testset "structure" begin
        b = kinked_breaks(-2.0, 6.0, [0.0]; n=24)
        @test issorted(b)
        @test length(b) == 25                       # n distinct + 1 repeat
        @test b[1] == -2.0 && b[end] == 6.0
        @test count(==(0.0), b) == 2                # default multiplicity
        @test length(unique(b)) == 24

        # multiplicity
        @test count(==(0.0), kinked_breaks(-2.0, 6.0, [0.0]; n=24, mult=1)) == 1
        @test count(==(0.0), kinked_breaks(-2.0, 6.0, [0.0]; n=24, mult=3)) == 3
        @test length(kinked_breaks(-2.0, 6.0, [0.0]; n=24, mult=3)) == 26

        # no kinks: evenly spaced
        @test kinked_breaks(0.0, 1.0, (); n=11) ≈ collect(range(0.0, 1.0, length=11))

        # several kinks, points spread in proportion to the piece lengths
        b = kinked_breaks(0.0, 10.0, [1.0, 5.0]; n=21)
        @test length(unique(b)) == 21
        @test count(==(1.0), b) == 2 && count(==(5.0), b) == 2
        @test length(filter(x -> 1.0 <= x <= 5.0, unique(b))) == 9   # 4/10 of 20 + 1
        @test kinked_breaks(0.0, 1.0, [0.5]; n=3) == [0.0, 0.5, 0.5, 1.0]
    end

    @testset "argument checks" begin
        @test_throws ArgumentError kinked_breaks(1.0, 0.0, [0.5]; n=10)
        @test_throws ArgumentError kinked_breaks(0.0, 1.0, [0.0]; n=10)   # at endpoint
        @test_throws ArgumentError kinked_breaks(0.0, 1.0, [1.5]; n=10)   # outside
        @test_throws ArgumentError kinked_breaks(0.0, 1.0, [0.5, 0.5]; n=10)
        @test_throws ArgumentError kinked_breaks(0.0, 1.0, [0.5]; n=10, mult=0)
        @test_throws ArgumentError kinked_breaks(0.0, 1.0, [0.5]; n=2)
    end

    # h(x) = max(x, 0)^2 is C^1 with a jump in h'' at 0 and piecewise
    # quadratic, so a cubic spline with a double breakpoint at 0 represents
    # it exactly (interpolation error at machine precision), while a spline
    # with a simple or no breakpoint at 0 cannot.
    @testset "interpolation of a C^1 kink" begin
        lb, ub = -2.0, 6.0
        h(x) = max(x, 0.0)^2
        x_test = collect(range(lb, ub, length=2001))

        function interp_error(breaks)
            basis = Basis(SplineParams(breaks, 0, 3))
            S, _ = nodes(basis)
            Phi = BasisMatrix(basis, Expanded(), S).vals[1]
            C = Phi \ h.(vec(S))
            fitted = vec(funeval(C, basis, reshape(x_test, :, 1)))
            return maximum(abs.(fitted .- h.(x_test)))
        end

        n = 24
        err_double = interp_error(kinked_breaks(lb, ub, [0.0]; n=n, mult=2))
        err_simple = interp_error(kinked_breaks(lb, ub, [0.0]; n=n, mult=1))
        err_even = interp_error(kinked_breaks(lb, ub, (); n=n))
        @test err_double < 1e-10
        @test err_simple > 1e-4
        @test err_even > 1e-4
    end

    # A basis with repeated breakpoints must work through the solver, with
    # the default first-order-condition inner solver, and not disturb a smooth
    # problem: log-utility growth with full depreciation has the closed-form
    # policy x = alpha * beta * s^alpha.
    @testset "solve with repeated breakpoints" begin
        s_min, s_max = 0.1, 2.0
        alpha, beta = 0.65, 0.95
        f(s, x) = log(s^alpha - x)
        g(s, x, e) = clamp(x, s_min, s_max)
        x_lb(s) = s_min
        x_ub(s) = min(s_max, s^alpha - 1e-6)
        cdp = ContinuousDP(f, g, beta, [0.0], [1.0], x_lb, x_ub)

        s_test = collect(range(0.3, 1.8, length=50))
        x_star = alpha * beta .* s_test .^ alpha

        for mult in (1, 2, 3), method in (PFI, VFI)
            breaks = kinked_breaks(s_min, s_max, [1.0]; n=15, mult=mult)
            basis = Basis(SplineParams(breaks, 0, 3))
            res = solve(cdp, CollocationSolver(basis; algorithm=method),
                        verbose=0)
            @test res.converged
            pf = ContinuousDPs.PolicyFunction(res)
            @test maximum(abs.([pf(s) for s in s_test] .- x_star)) < 5e-3
        end
    end
end
