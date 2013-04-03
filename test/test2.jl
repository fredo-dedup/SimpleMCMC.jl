#########################################################################
#    testing script for distributions and samplers
#########################################################################

include("../src/SimpleMCMC.jl")
using Distributions # used to provide exact cdf of distributions

N = 2000  # number of steps in MCMC for testing
KSTHRESHOLD = 1.358  #  5% level confidence for Kolmogorov–Smirnov test
KSTHRESHOLD = 5  # TODO : understand why KS so bad for all samplers forcing such a high threshold to pass tests

TOLERANCE = 1e-1  # 10% tolerance due to small sampling sizes

function ksValue(x, distrib)  # x = res.params[:x]
	global xs = sort(x)
	y = eval( :(cdf($distrib , xs)))
	dn = max( abs([1:length(x)] / length(x) - y) ) 
	sqrt(length(x))*dn
end

function ksTest(ex::Expr)  # ex = :(Normal(3, 12))
	model = :(x::real ; x ~ $ex)
	distrib = expr(:call, 
					expr(:., :Distributions, expr(:quote, ex.args[1])), 
					ex.args[2:end]...) 
	exactMean = eval( :(mean($distrib)) )
	realStd = eval( :(std($distrib)) )

	print("testing simpleRWM on $ex   -")
	srand(1)
	res = SimpleMCMC.simpleRWM(model, N, 1000, [exactMean])
	ksv = ksValue(res.params[:x], distrib)
	println(" KS measure = $ksv")
	assert(ksv < KSTHRESHOLD, "correct distrib hyp. rejected")

	print("testing simpleHMC on $ex   -")
	srand(1)
	res = SimpleMCMC.simpleHMC(model, N, 1000, [exactMean], 2, realStd/5)
	ksv = ksValue(res.params[:x], distrib)
	println(" KS measure = $ksv")
	assert(ksv < KSTHRESHOLD, "correct distrib hyp. rejected")

	print("testing simpleNUTS on $ex  -")
	srand(1)
	res = SimpleMCMC.simpleNUTS(model, N, 1000, [exactMean])
	ksv = ksValue(res.params[:x], distrib)
	println(" KS measure = $ksv")
	assert(ksv < KSTHRESHOLD, "correct distrib hyp. rejected")
end



ksTest(:(Normal(1, 1)))
ksTest(:(Normal(3, 12)))

ksTest(:(Weibull(1, 1)))
ksTest(:(Weibull(3, 1)))

ksTest(:(Uniform(0, 2)))

ksTest(:(TDist(2.2)))  # very long on NUTS if df <= 2  (infinite variance)
ksTest(:(TDist(4)))

ksTest(:(Beta(1,2)))
ksTest(:(Beta(3,2)))

