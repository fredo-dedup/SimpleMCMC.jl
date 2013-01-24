#########################################################################
#    testing script
#########################################################################

	include("../src/SimpleMCMC.jl")

module Test

	include("../src/SimpleMCMC.jl")

	DIFF_DELTA = 1e-8
	ERROR_THRESHOLD = 1e-3

	ex = :(SimpleMCMC.logpdfUniform(-1, x, 0))
	x0 = 0.5
	############# derivative checking function  ######################

	function test1(ex::Expr, x0::Union(Float64, Vector{Float64}, Matrix{Float64})) #  ex= :(2*x) ; x0 = 2.
		model = :(x::real; y = $ex; y ~ TestDiff())

		myf, np = Test.SimpleMCMC.buildFunctionWithGradient(model)
		ex2 = myf.args[2].args

		cex = expr(:block, vcat({:(__beta = [$x0])}, ex2))
		l0, grad0 = eval(cex)  #  __loglik([x0])
		print("$ex  at $l0   ===== $(round(grad0,5)) =====")

		cex = expr(:block, vcat({:(__beta = [$(x0+DIFF_DELTA)])}, ex2))
		# println("<<<<\n$cex\n>>>>")
		l, grad = eval(cex) # __loglik([x0] + DIFF_DELTA)
		gradn = (l-l0)/DIFF_DELTA
		println("  $(round(gradn,5))  ======")

		assert(abs(gradn - grad0) < ERROR_THRESHOLD, 
			"Gradient false for $ex at x=$x0, expected $(round(gradn,5)), got $(round(grad0,5))")
	end

	test1(e::Expr, v::Vector{Any}) = (for x in v; test1(e, x);end)

end

################################

Test.test1( :(2+x), {-1., 0., 1., 10.})
Test.test1( :(x+4), {-1., 0., 1., 10.})

Test.test1( :(2-x), {-1., 0., 1., 10.})
Test.test1( :(x-2), {-1., 0., 1., 10.})
Test.test1( :(-x),  {-1., 0., 1., 10.})

Test.test1( :(2*x), {-1., 0., 1., 10.})
Test.test1( :(x*2), {-1., 0., 1., 10.})

Test.test1( :(2/x), {-1., 0.1, 1., 10.})
Test.test1( :(x/2), {-1., 0., 1., 10.})

Test.test1(	:(sum(x)), 	{-1., 0., 1., 10.})

# Test.test1(	:(dot(x,2)), 	{-1., 0., 1., 10.})
# Test.test1(	:(dot(2,x)), 	{-1., 0., 1., 10.})

Test.test1(	:(x.*2), {-1., 0., 1., 10.})
Test.test1(	:(2.*x), {-1., 0., 1., 10.})

Test.test1(	:(log(x)), {.01, 1., 10.})
Test.test1(:(exp(x)), {-1., 0., 1., 3.})

Test.test1(:(sin(x)), {-1., 0., 1., 10.})
Test.test1(:(cos(x)), {-1., 0., 1., 10.})

Test.test1(:(SimpleMCMC.logpdfNormal(0, 1, x)), {-1., 0., 1.})
Test.test1(:(SimpleMCMC.logpdfNormal(0, x, 1)), {0.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfNormal(x, 1, 0)), {-1., 0., 1.})

Test.test1(:(SimpleMCMC.logpdfWeibull(2, 1, x)), {.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfWeibull(2, x, 1)), {0.5, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfWeibull(x, 1, 0.5)), {0.5, 1., 10.})

Test.test1(:(SimpleMCMC.logpdfUniform(-1, 1, x)), {-.1, 0., 0.9})
Test.test1(:(SimpleMCMC.logpdfUniform(-1, x, 0)), {0.5, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfUniform(x, 10, 8.)), {-0.5, 1., 5.})


    __beta = [0.2]
    x = __beta[1]
    __acc = 2.0
    y = SimpleMCMC.logpdfUniform(-1, x, 0)
    __tmp_1032 = SimpleMCMC.logpdfTestDiff(y)
    ____acc_1033 = +(__acc, __tmp_1032)
    d____acc_1033 = 2.0
    dy = zero(y)
    d__tmp_1032 = zero(__tmp_1032)
    dx = zero(x)
    d__tmp_1032 += d____acc_1033
    dy += d__tmp_1032
    dx += log(./(-(.*([(-1.<=0.<=x)], 2.0)), .^(-(x, -1), 2.0)))
    (____acc_1033, dx)


# Test.test1(:(SimpleMCMC.logpdfUniform(-1, 1, x)), 1.)


ex = :(SimpleMCMC.logpdfUniform(-1, 1, x))
model = :(x::real; y = $ex; y ~ TestDiff())

myf, np = Test.SimpleMCMC.buildFunctionWithGradient(model)
ex2 = myf.args[2].args
# x0 = 1.
# cex = expr(:block, vcat({:(__beta = [$x0])}, ex2))

# Test.SimpleMCMC.logpdf(Test.SimpleMCMC.Uniform(-1., 1.), 1.1)


    __beta = [1.0000001]
    x = __beta[1]
    __acc = 0.0
    y = Test.SimpleMCMC.logpdfUniform(-1, x, x)
    __tmp_23 = Test.SimpleMCMC.logpdfTestDiff(y)
    __tmp_24 = sum(__tmp_23)    #  <= le pb est là !  sum(Inf) ne retourne jamais
    ____acc_25 = +(__acc, __tmp_23)
    d____acc_25 = 1.0
    dy = zero(y)
    d__tmp_24 = 0.0
    dx = zero(x)
    d__tmp_23 = zero(__tmp_23)
    d__tmp_24 += d____acc_25
    d__tmp_23 += ./(d__tmp_24, length(__tmp_23))
    dy += d__tmp_23
    dx += 0.0
    (____acc_25, dx)



# println("<<<<\n$cex\n>>>>")

# Test.SimpleMCMC.logpdfUniform(-1, 1, -10.)

# [ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]
