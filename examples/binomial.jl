######### logistic regression on 1000 obs x 10 var  ###########

include("../src/SimpleMCMC.jl")

using Distributions

# simulate dataset
begin
	srand(1)
	n = 1000
	nbeta = 10 # number of predictors, including intercept
	X = [ones(n) randn((n, nbeta-1))]
	beta0 = randn((nbeta,))
	Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0 + randn((n,)) ) ))
end

# define model
model = quote
	vars::vector(nbeta)

	vars ~ Normal(0, 1.0)  # Normal prior, variance 1.0 for predictors
	resid = Y - ( 1 / (1. + exp(X * vars)) )
	resid ~ Normal(0, 1.0)  
end

# run random walk metropolis (10000 steps, 500 for burnin)
res = SimpleMCMC.simpleRWM(model, 1000)

mean(res[:,2]) # accept rate
[ [mean(res[:,i+2])::Float64 for i in 1:nbeta] beta0 ] # calculated and original values side by side


# run Hamiltonian Monte-Carlo (1000 steps, 500 for burnin, 5 inner steps, 0.01 inner step size)
res = SimpleMCMC.simpleHMC(model, 1000, 5, 1e-2)

mean(res[:,2]) # accept rate
[ [mean(res[:,i])::Float64 for i in 3:size(res,2)] beta0 ] # calculated and original values side by side



