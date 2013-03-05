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
	# res = SimpleMCMC.simpleRWM(model, 10, 0, [realMean])
	println(res.params[:x])

	calcMean = mean(res.params[:x])
	calcStd = std(res.params[:x])
	assert(abs((calcMean-realMean)/realStd) < TOLERANCE, "expected mean $realMean, got $calcMean")
	assert(abs((calcStd-realStd)/realStd) < TOLERANCE, "expected std $realStd, got $calcStd")

	println("testing simpleHMC x $ex")
	srand(1)
	res = SimpleMCMC.simpleHMC(model, 100000, 1000, [realMean], 2, realStd/5)
	# res = SimpleMCMC.simpleHMC(model, 10, 0, [realMean], 2, realStd/5)
	
	calcMean = mean(res.params[:x])
	calcStd = std(res.params[:x])
	assert(abs((calcMean-realMean)/realStd) < TOLERANCE, "expected mean $realMean, got $calcMean")
	assert(abs((calcStd-realStd)/realStd) < TOLERANCE, "expected std $realStd, got $calcStd")

	println("testing simpleNUTS x $ex")
	srand(1)
	res = SimpleMCMC.simpleNUTS(model, 10000, 1000, [realMean])
	println(res.params[:x])

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


# z = randn(10) .< 0.5
# samplers1(:(Normal(1, 1)))

# model = :(x::real ; x ~ Normal(0,1))
# res = SimpleMCMC.simpleRWM(model, 100000, 1000, 0.0)
# res = SimpleMCMC.simpleHMC(model, 100000, 1000, 0.0, 2, 0.2)

# model = :(x::real ; x ~ Normal(3,1))
# res = SimpleMCMC.simpleRWM(model, 100000, 1000, 0.0)
# res = SimpleMCMC.simpleRWM(model, 10, 0, 0.0)
# res = SimpleMCMC.simpleHMC(model, 100000, 1000, 0.0, 2, 0.2)

# model = :(x::real ; x ~ Weibull(1,1))
# res = SimpleMCMC.simpleRWM(model, 100000, 1000, 1.0)
# res = SimpleMCMC.simpleRWM(model, 10, 0, 1.0)
# res = SimpleMCMC.simpleHMC(model, 100000, 1000, 1.0, 2, 0.2)
# srand(1)
# res = SimpleMCMC.simpleHMC(model, 10, 0, 1.0, 2, 0.2)

# __loglik([1.1])

# myf, n = SimpleMCMC.buildFunctionWithGradient(model)

# # model = :(x::real ; z ~ Bernoulli(x)) # mean 0.5, std ...
# # recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.5]))  # 6.200 ess/s
# # recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.5], 2, 0.04)) # 140 ess/s
# # recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.5]))  # 17.000 ess/s, correct



# #########################

# include("../src/SimpleMCMC.jl")
# function samplers2(ex::Expr)  # ex = :(Normal(3, 12))
# 	model = :(x::real ; x ~ $ex)

# 	res = SimpleMCMC.simpleRWM(model, 10000, 0, [1.])
# 	println("ll : $(res[end,1]), x : $(res[end,3])")

# 	res = SimpleMCMC.simpleHMC(model, 10000, 0, [1.], 2, 1./5)
# 	println("ll : $(res[end,1]), x : $(res[end,3])")
# end

# samplers2(:(Weibull(1, 1)))
# samplers1(:(Weibull(1, 1)))

# SimpleMCMC.mwhos()

# __loglik([-14.])


# model = :(x::real ; x ~ Weibull(1,1))
# res = SimpleMCMC.simpleRWM(model, 100000, 1000, 1.0)
# res = SimpleMCMC.simpleRWM(model, 10, 0, 1.0)


# whos()
# whos(SimpleMCMC)
