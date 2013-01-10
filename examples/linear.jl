######### linear regression 1000 obs x 10 var  ###########

load("simple-mcmc/src/SimpleMCMC.jl")
using SimpleMCMC

require("Distributions.jl/src/distributions.jl")
using Distributions

# simulate dataset
begin
	srand(1)
	n = 1000
	nbeta = 10 # number of predictors, including intercept
	X = [ones(n) randn((n, nbeta-1))]
	beta0 = randn((nbeta,))
	Y = X * beta0 + randn((n,))
end

# define model
model = quote
	vars::vector(nbeta)

	vars ~ Normal(0, 1.0)  # Normal prior, variance 1.0 for predictors
	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  
end

# run random walk metropolis (1000 steps, 500 for burnin)
res = simpleRWM(model, 1000)

# calculated parameters and original values side by side
[ [mean(res[:,i])::Float64 for i in 3:size(res,2)] beta0 ]


# run Hamiltonian Monte-Carlo (1000 steps, 500 for burnin, 2 inner steps, 0.1 inner step size)
res = SimpleMCMC.simpleHMC(model, 1000, 2, 0.1)
SimpleMCMC.buildFunctionWithGradient(model)

# calculated parameters and original values side by side
[ [mean(res[:,i])::Float64 for i in 3:size(res,2)] beta0 ]



