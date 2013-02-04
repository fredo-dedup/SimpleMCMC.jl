######### linear regression 1000 obs x 10 var  ###########

include("../src/SimpleMCMC.jl")

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
	vars::real(nbeta)

	vars ~ Normal(0, 1.0)  # Normal prior, variance 1.0 for predictors
	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  
end

# run random walk metropolis (10000 steps, 5000 for burnin)
res = SimpleMCMC.simpleRWM(model, 10000)

[ [mean(res[:,i+2])::Float64 for i in 1:nbeta] beta0 ] # show original values and mean of samples side by side


# run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 8 inner steps, 0.01 inner step size)
res = SimpleMCMC.simpleHMC(model, 10000, 1000, 8, 0.01)

[ [mean(res[:,i+2])::Float64 for i in 1:nbeta] beta0 ] # show original values and mean of samples side by side

# run NUTS flavoured HMC (1000 steps, 500 for burnin)
res = SimpleMCMC.simpleNUTS(model, 1000)

[ [mean(res[:,i+2])::Float64 for i in 1:nbeta] beta0 ] # show original values and mean of samples side by side

