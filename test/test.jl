#########################################################################
#    testing script
#########################################################################

require("../src/SimpleMCMC.jl")

const DIFF_DELTA = 1e-10
const ERROR_THRESHOLD = 1e-5

l0, grad = __loglik(ones(2))
[ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]

############# derivative checking function  ######################

function test1(e::Expr, x0::Float64)
	model = :(x::real; y = $e; y ~ TestDiff())

	myf, np = SimpleMCMC.buildFunctionWithGradient(model)
	eval(myf)

	l0, grad0 = __loglik(ones(1)*x0)
	l, grad = __loglik(ones(1)*x0 + DIFF_DELTA)
	assert(abs((l-l0)/delta - grad0) < ERROR_THRESHOLD, "Diff error on expression $e")
end

test1(e::Expr) = test1(e, 1.0)

################################

test1(:(2*x))
test1(:(2*x), 2.)

test1(:(x^2))
test1(:(x^2), 4.)
test1(:(x^2), -1.)

test1(:(2^x))
test1(:(2^x), )
test1(:(2^x))
test1(:(2^x))
test1(:(2+x))
test1(:(x+2))
test1(:(log(x)))
test1(:(exp(x)))
test1(:(sin(x)))
test1(:(cos(x)))


[ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]
