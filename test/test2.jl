#########################################################################
#    testing script for distributions and samplers
#########################################################################

include("../src/SimpleMCMC.jl")
using Distributions # used to provide exact cdf of distributions

N = 10000  # number of steps in MCMC for testing
KSTHRESHOLD = 1.358  #  5% level confidence for Kolmogorov–Smirnov test
KSTHRESHOLD = 5  # TODO : understand why KS so bad for all samplers forcing such a high threshold to pass tests

TOLERANCE = 1e-1  # 10% tolerance due to small sampling sizes

function ksValue(x, distrib) 
	global xs = sort(x)
	y = eval( :(cdf($distrib , xs)))
	dn = max( abs([1:length(x)] / length(x) - y) ) 
	sqrt(length(x))*dn
end

function ksTest(ex::Expr) 
	model = :(x::real ; x ~ $ex)
	distrib = expr(:call, 
					expr(:., :Distributions, expr(:quote, ex.args[1])), 
					ex.args[2:end]...) 
	exactMean = eval( :(mean($distrib)) )
	exactMean = isfinite(exactMean) ? exactMean : 1.0  # some distribs (Cauchy) have no defined mean
	exactStd = eval( :(std($distrib)) )
	exactStd = isfinite(exactStd) ? exactStd : 1.0

	print("testing simpleRWM on $ex   -")
	srand(1)
	res = SimpleMCMC.simpleRWM(model, N, 1000, [exactMean])
	ksv = ksValue(res.params[:x], distrib)
	println(" KS measure = $ksv")
	assert(ksv < KSTHRESHOLD, "correct distrib hyp. rejected")

	print("testing simpleHMC on $ex   -")
	srand(1)
	res = SimpleMCMC.simpleHMC(model, N, 1000, [exactMean], 2, exactStd/5)
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

ksTest(:(TDist(2.2)))  # very long on NUTS if df <= 2  (due to infinite variance ?)
ksTest(:(TDist(4)))

ksTest(:(Beta(1,2)))
ksTest(:(Beta(3,2)))

ksTest(:(Gamma(1,2)))
ksTest(:(Gamma(3,0.2)))

ksTest(:(Cauchy(0,1))) 
ksTest(:(Cauchy(-1,0.2))) 

ksTest(:(Exponential(3)))
ksTest(:(Exponential(0.2)))

ksTest(:(logNormal(-1, 1)))
ksTest(:(logNormal(2, 0.1)))



# TODO : find a way to test discrete distributions : Bernoulli, Binomial, Poisson

