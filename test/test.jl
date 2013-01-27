#########################################################################
#    testing script
#########################################################################

include("../src/SimpleMCMC.jl")

module Test

	include("../src/SimpleMCMC.jl")

	DIFF_DELTA = 1e-10
	ERROR_THRESHOLD = 1e-2

	############# derivative checking function  ######################

	function test1(ex::Expr, x0::Union(Float64, Vector{Float64}, Matrix{Float64})) #  ex= :(2*x) ; x0 = 2.
		println("testing derivation of $ex")

		model = :(x::real; y = $ex; y ~ TestDiff())

		myf, np = Test.SimpleMCMC.buildFunctionWithGradient(model)
		ex2 = myf.args[2].args

		cex = expr(:block, vcat({:(__beta = [$x0])}, ex2))
		l0, grad0 = Main.eval(cex)  #  __loglik([x0])
		# print("$ex  at $l0   ===== $(round(grad0,5)) =====")

		cex = expr(:block, vcat({:(__beta = [$(x0+DIFF_DELTA)])}, ex2))
		# println("<<<<\n$cex\n>>>>")
		l, grad = Main.eval(cex) # __loglik([x0] + DIFF_DELTA)
		gradn = (l-l0)/DIFF_DELTA
		# println("  $(round(gradn,5))  ======")

        delta = abs(gradn - grad0) ./ max([abs(grad0)], 0.01)
		assert(max([delta]) < ERROR_THRESHOLD, 
			"Gradient false for $ex at x=$x0, expected $(round(gradn,5)), got $(round(grad0,5))")
	end

	test1(e::Expr, v::Vector{Any}) = (map(x->test1(e, x), v); nothing)

end

######### real var derivation with other real arguments #######################
Test.test1( :(2+x), {-1., 0., 1., 10.})
Test.test1( :(x+4), {-1., 0., 1., 10.})

Test.test1( :(2-x), {-1., 0., 1., 10.})
Test.test1( :(x-2), {-1., 0., 1., 10.})
Test.test1( :(-x),  {-1., 0., 1., 10.})

Test.test1( :(2*x), {-1., 0., 1., 10.})
Test.test1( :(x*2), {-1., 0., 1., 10.})

Test.test1( :(2/x), {-1., 0.1, 1., 10.})
Test.test1( :(x/2), {-1., 0., 1., 10.})

Test.test1(:(sum(x)),{-1., 0., 1., 10.})

Test.test1(:(x.*2), {-1., 0., 1., 10.})
Test.test1(:(2.*x), {-1., 0., 1., 10.})

Test.test1(:(log(x)), {.01, 1., 10.})
Test.test1(:(exp(x)), {-1., 0., 1., 3.})

Test.test1(:(sin(x)), {-1., 0., 1., 10.})
Test.test1(:(cos(x)), {-1., 0., 1., 10.})

Test.test1(:(SimpleMCMC.logpdfNormal(1, 2, x)), {-1., 0., 1.})
Test.test1(:(SimpleMCMC.logpdfNormal(-1, x, 1)), {0.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfNormal(x, 2, 0)), {-1., 0., 1.})

Test.test1(:(SimpleMCMC.logpdfWeibull(2, 1, x)), {.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfWeibull(2, x, 1)), {0.5, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfWeibull(x, 1, 0.5)), {0.5, 1., 10.})

Test.test1(:(SimpleMCMC.logpdfUniform(-1, 1, x)), {-.1, 0., 0.9})
Test.test1(:(SimpleMCMC.logpdfUniform(0, x, 0)), {0.5, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfUniform(x, 0.5, 0.2)), {0., -1., -5.})

######### real var derivation with other vector arguments #######################
z = [2., 3, 0.1]

Test.test1( :(z+x), {-1., 0., 1., 10.})
Test.test1( :(x+z), {-1., 0., 1., 10.})

Test.test1( :(z-x), {-1., 0., 1., 10.})
Test.test1( :(x-z), {-1., 0., 1., 10.})

# Test.test1( :(z*x), {-1., 0., 1., 10.})
# Test.test1( :(x*z), {-1., 0., 1., 10.})

Test.test1( :(z/x), {-1., 0.1, 1., 10.})
Test.test1( :(x/z), {-1., 0., 1., 10.})

# Test.test1( :(x.*z), {-1., 0., 1., 10.}) # ERROR
# Test.test1( :(z.*x), {-1., 0., 1., 10.}) # ERROR

Test.test1(:(SimpleMCMC.logpdfNormal(z, 1, x)), {-1., 0., 1.})
Test.test1(:(SimpleMCMC.logpdfNormal(0., z, x)), {-1., 0., 1.})
Test.test1(:(SimpleMCMC.logpdfNormal(z, z, x)), {-1., 0., 1.})

Test.test1(:(SimpleMCMC.logpdfNormal(0, x, z)), {0.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfNormal(z, x, 0)), {0.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfNormal(z, x, z)), {0.1, 1., 10.})

Test.test1(:(SimpleMCMC.logpdfNormal(x, 1, z)), {-1., 0., 1.})
Test.test1(:(SimpleMCMC.logpdfNormal(x, z, 1)), {-1., 0., 1.})
Test.test1(:(SimpleMCMC.logpdfNormal(x, z, 1)), {-1., 0., 1.})


Test.test1(:(SimpleMCMC.logpdfWeibull(z, 1, x)), {.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfWeibull(2, z, x)), {.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfWeibull(z, z, x)), {.1, 1., 10.})

Test.test1(:(SimpleMCMC.logpdfWeibull(z, x, 1)), {.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfWeibull(2, x, z)), {.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfWeibull(z, x, z)), {.1, 1., 10.})

Test.test1(:(SimpleMCMC.logpdfWeibull(x, z, 1)), {.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfWeibull(x, 1, z)), {.1, 1., 10.})
Test.test1(:(SimpleMCMC.logpdfWeibull(x, z, z)), {.1, 1., 10.})


Test.test1(:(SimpleMCMC.logpdfUniform(z, 20, x)), {4., 9., 10.})
Test.test1(:(SimpleMCMC.logpdfUniform(-10, z, x)), {-.1, -1., 0.})
Test.test1(:(SimpleMCMC.logpdfUniform(z, z+5, x)), {3.1, 5., 4.})

Test.test1(:(SimpleMCMC.logpdfUniform(z, x, 4)), {5.1, 9., 10.})
Test.test1(:(SimpleMCMC.logpdfUniform(-5, x, z)), {4.1, 5., 10.})
Test.test1(:(SimpleMCMC.logpdfUniform(z, x, z+1.)), {6.1, 7., 10.})

Test.test1(:(SimpleMCMC.logpdfUniform(x, z, 0)), {-.1, -1., -10.})
Test.test1(:(SimpleMCMC.logpdfUniform(x, 5, z)), {0., -1., -10.})
Test.test1(:(SimpleMCMC.logpdfUniform(x, z, z-1.)), {-2.1, -3., -10.})

# [ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]



######### vector var derivation with other real arguments #######################
z = [2., 3, 0.1]



