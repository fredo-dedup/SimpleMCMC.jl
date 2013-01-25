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
Test.test1(:(SimpleMCMC.logpdfUniform(0, x, 0)), {0.5, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfUniform(x, 0.5, 0.2)), {0., 1., 5.})



# [ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]
