#########################################################################
#    testing script
#########################################################################

require("../src/SimpleMCMC.jl")

DIFF_DELTA = 1e-8
ERROR_THRESHOLD = 1e-3

############# derivative checking function  ######################

function test1(ex::Expr, x0::Float64) #  ex= :(2*x) ; x0 = 2.
	model = :(x::real; y = $ex; y ~ TestDiff())

	myf, np = SimpleMCMC.buildFunctionWithGradient(model)
	ex2 = myf.args[2].args

	cex = expr(:block, vcat({:(__beta = [$x0])}, ex2))
	# println("<<<<\n$cex\n>>>>")
	l0, grad0 = eval(cex)  #  __loglik([x0])
	print("=====  $l0   ===== $grad0 =====")

	cex = expr(:block, vcat({:(__beta = [$(x0+DIFF_DELTA)])}, ex2))
	# println("<<<<\n$cex\n>>>>")
	l, grad = eval(cex) # __loglik([x0] + DIFF_DELTA)
	gradn = (l-l0)/DIFF_DELTA
	println("  $gradn  ======")
	assert(abs(gradn - grad0) < ERROR_THRESHOLD, "Diff error on expression $ex at x=$x0")
end

test1(e::Expr, v::Vector{Float64}) = (for x in v; test1(e, x);end)

################################

__loglik = nothing
test1(:(2+x), [-1., 0, 1, 10])
test1(:(x+4), [-1., 0, 1, 10])

test1(:(2-x), [-1., 0, 1, 10])
test1(:(x-2), [-1., 0, 1, 10])
test1(:(-x), [-1., 0, 1, 10])

test1(:(2*x), [-1., 0, 1, 10])
test1(:(x*2), [-1., 0, 1, 10])

test1(:(2/x), [-1., 0.1, 1, 10])
test1(:(x/2), [-1., 0, 1, 10])

test1(:(sum(x,2)), [-1., 0, 1, 10])
test1(:(sum(2,x)), [-1., 0, 1, 10])

test1(:(dot(x,2)), [-1., 0, 1, 10])
test1(:(dot(2,x)), [-1., 0, 1, 10])

test1(:(x.*2), [-1., 0, 1, 10])
test1(:(2.*x), [-1., 0, 1, 10])

test1(:(log(x)), [0.001, 1, 10])
test1(:(exp(x)), [-1., 0, 1, 10])

res = __loglik([-10.])
dump(res)
res[2]
exp(2)

test1(:(sin(x)), [-1., 0, 1])
test1(:(cos(x)), [-1., 0, 1])

test1(:(sin(x)), [-1., 0, 1])
test1(:(cos(x)), [-1., 0, 1])

test1(:(SimpleMCMC.logpdfNormal(0, 1, x)), [-1., 0, 1])
test1(:(SimpleMCMC.logpdfNormal(0, x, 1)), [0.5, 1, 10])
test1(:(SimpleMCMC.logpdfNormal(x, 1, 0)), [-1., 1, 10])

test1(:(SimpleMCMC.logpdfWeibull(0, 1, x)), [-1., 0, 1])
test1(:(SimpleMCMC.logpdfNormal(0, x, 1)), [0.5, 1, 10])
test1(:(SimpleMCMC.logpdfNormal(x, 1, 0)), [-1., 1, 10])

SimpleMCMC.logpdfWeibull(0, 1, -1.)

# [ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]
