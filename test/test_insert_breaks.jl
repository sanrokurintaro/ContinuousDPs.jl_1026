using BasisMatrices: BasisMatrix, Expanded, funeval

@testset "insert_breaks" begin
    even(lb, ub, n, k=3) = SplineParams(range(lb, ub, length=n), 0, k)

    @testset "SplineParams" begin
        p = even(-2.0, 6.0, 24)
        @test p.evennum == 0                        # fixture: explicit breaks

        q = insert_breaks(p, [0.0]; multiplicity=2)
        b = q.breaks
        @test q isa SplineParams && q.k == 3 && q.evennum == 0
        @test issorted(b) && b[1] == -2.0 && b[end] == 6.0
        @test length(b) == 26                       # 24 + 2 inserted
        @test count(==(0.0), b) == 2

        # default multiplicity is a simple breakpoint; the input is untouched
        @test count(==(0.0), insert_breaks(p, [0.0]).breaks) == 1
        @test length(p.breaks) == 24
        @test count(==(0.0), insert_breaks(p, [0.0]; multiplicity=3).breaks) == 3

        # a location that is already a breakpoint is topped up, never reduced
        q = insert_breaks(p, [p.breaks[5]]; multiplicity=2)
        @test count(==(p.breaks[5]), q.breaks) == 2 && length(q.breaks) == 25
        @test insert_breaks(q, [p.breaks[5]]; multiplicity=1).breaks == q.breaks

        # no locations: unchanged; evenly spaced input gets explicit breaks
        @test insert_breaks(p, ()) === p
        pe = SplineParams([0.0, 1.0], 11, 3)
        @test pe.evennum == 11
        qe = insert_breaks(pe, [0.5]; multiplicity=2)
        @test qe.evennum == 0 && length(qe.breaks) == 12
        @test count(==(0.5), qe.breaks) == 2        # already one; topped up

        # several locations
        q = insert_breaks(SplineParams([0.0, 10.0], 0, 3), [5.0, 1.0];
                          multiplicity=2)
        @test q.breaks == [0.0, 1.0, 1.0, 5.0, 5.0, 10.0]
    end

    @testset "LinParams" begin
        p = LinParams(collect(range(0.0, 1.0, length=5)), 0)
        q = insert_breaks(p, [0.1, 0.5])
        @test q isa LinParams && q.evennum == 0
        @test q.breaks == [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]   # 0.5 not repeated
        @test insert_breaks(p, ()) === p
    end

    @testset "Basis" begin
        sp = even(-2.0, 6.0, 24)
        cp = ChebParams(7, -0.5, 0.5)
        basis = Basis(sp, cp)

        nb = insert_breaks(basis, 1, [0.0]; multiplicity=2)
        @test nb isa Basis{2}
        @test nb.params[1].breaks == insert_breaks(sp, [0.0]; multiplicity=2).breaks
        @test nb.params[2] === cp
        @test basis.params[1] === sp

        # one-dimensional basis, piecewise linear dimension
        lb = Basis(LinParams([0.0, 1.0], 0))
        @test insert_breaks(lb, 1, [0.5]).params[1].breaks == [0.0, 0.5, 1.0]
        @test_throws ArgumentError insert_breaks(lb, 1, [0.5]; multiplicity=2)

        @test_throws ArgumentError insert_breaks(basis, 2, [0.0])   # Chebyshev
        @test_throws ArgumentError insert_breaks(basis, 3, [0.0])   # no such dim
        @test_throws ArgumentError insert_breaks(basis, 0, [0.0])
    end

    @testset "argument checks" begin
        p = even(0.0, 1.0, 10)
        @test_throws ArgumentError insert_breaks(p, [0.0])          # at endpoint
        @test_throws ArgumentError insert_breaks(p, [1.0])
        @test_throws ArgumentError insert_breaks(p, [1.5])          # outside
        @test_throws ArgumentError insert_breaks(p, [0.5, 0.5])     # not distinct
        @test_throws ArgumentError insert_breaks(p, [0.5]; multiplicity=0)
        @test_throws ArgumentError insert_breaks(p, [0.5]; multiplicity=4)  # > k
        @test_throws ArgumentError insert_breaks(LinParams([0.0, 1.0], 0), [2.0])
    end

    # h(x) = max(x, 0)^2 is C^1 with a jump in h'' at 0 and piecewise
    # quadratic, so a cubic spline with a double breakpoint at 0 represents
    # it exactly (interpolation error at machine precision), while a spline
    # with a simple or no breakpoint at 0 cannot.
    @testset "interpolation of a C^1 kink" begin
        lb, ub = -2.0, 6.0
        h(x) = max(x, 0.0)^2
        x_test = collect(range(lb, ub, length=2001))

        function interp_error(p)
            basis = Basis(p)
            S, _ = nodes(basis)
            Phi = BasisMatrix(basis, Expanded(), S).vals[1]
            C = Phi \ h.(vec(S))
            fitted = vec(funeval(C, basis, reshape(x_test, :, 1)))
            return maximum(abs.(fitted .- h.(x_test)))
        end

        # even breakpoints that do not include 0; 0 is then inserted
        p = even(lb, ub, 24)
        @test !(0.0 in p.breaks)
        err_double = interp_error(insert_breaks(p, [0.0]; multiplicity=2))
        err_simple = interp_error(insert_breaks(p, [0.0]; multiplicity=1))
        err_even = interp_error(p)
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
            basis = insert_breaks(Basis(even(s_min, s_max, 14)), 1, [1.0];
                                  multiplicity=mult)
            res = solve(cdp, CollocationSolver(basis; algorithm=method),
                        verbose=0)
            @test res.converged
            pf = ContinuousDPs.PolicyFunction(res)
            @test maximum(abs.([pf(s) for s in s_test] .- x_star)) < 5e-3
        end
    end
end
