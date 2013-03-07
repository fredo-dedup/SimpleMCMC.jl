#########################################################################
#    testing script for distributions and samplers
#########################################################################

include("../src/SimpleMCMC.jl")
require("Distributions")

# using Distributions  # used to provide exact mean and std of distributions

TOLERANCE = 1e-1  # 10% tolerance due to small sampling sizes

function samplers1(ex::Expr)  # ex = :(Normal(3, 12))
	model = :(x::real ; x ~ $ex)
	distrib = expr(:call, 
					expr(:., :Distributions, expr(:quote, ex.args[1])), 
					ex.args[2:end]...) 
	realMean = eval( :(mean($distrib)) )
	realStd = eval( :(std($distrib)) )

	println("testing simpleRWM x $ex")
	srand(1)
	res = SimpleMCMC.simpleRWM(model, 100000, 1000, [realMean])

	calcMean = mean(res.params[:x])
	calcStd = std(res.params[:x])
	assert(abs((calcMean-realMean)/realStd) < TOLERANCE, "expected mean $realMean, got $calcMean")
	assert(abs((calcStd-realStd)/realStd) < TOLERANCE, "expected std $realStd, got $calcStd")

	println("testing simpleHMC x $ex")
	srand(1)
	res = SimpleMCMC.simpleHMC(model, 100000, 1000, [realMean], 2, realStd/5)
	
	calcMean = mean(res.params[:x])
	calcStd = std(res.params[:x])
	assert(abs((calcMean-realMean)/realStd) < TOLERANCE, "expected mean $realMean, got $calcMean")
	assert(abs((calcStd-realStd)/realStd) < TOLERANCE, "expected std $realStd, got $calcStd")

	println("testing simpleNUTS x $ex")
	srand(1)
	res = SimpleMCMC.simpleNUTS(model, 10000, 1000, [realMean])

	calcMean = mean(res.params[:x])
	calcStd = std(res.params[:x])
	assert(abs((calcMean-realMean)/realStd) < TOLERANCE, "expected mean $realMean, got $calcMean")
	assert(abs((calcStd-realStd)/realStd) < TOLERANCE, "expected std $realStd, got $calcStd")
end

samplers1(:(Weibull(1, 1)))
samplers1(:(Weibull(3, 1)))
samplers1(:(Uniform(0, 2)))
samplers1(:(Normal(1, 1)))
samplers1(:(Normal(3, 12)))

# TODO : find way to test Bernoulli distrib
# # model = :(x::real ; z ~ Bernoulli(x)) # mean 0.5, std ...
