######### linear regression 1000 obs x 10 var  ###########

include("../src/SimpleMCMC.jl")

# simulate dataset
srand(1)
n = 1000
nbeta = 10 # number of predictors, including intercept
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = X * beta0 + randn((n,))

# define model
model = quote
	vars::real(nbeta)

	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  
end

# run random walk metropolis (10000 steps, 5000 for burnin)
res = SimpleMCMC.simpleRWM(model, 10000)

res.acceptRate  # acceptance rate
[ sum(res.params[:vars],2)./res.samples beta0 ] # show calculated and original coefs side by side

# run Hamiltonian Monte-Carlo (1000 steps, 500 for burnin, 2 inner steps, 0.05 inner step size)
res = SimpleMCMC.simpleHMC(model, 1000, 2, 0.05)

# run NUTS - HMC (1000 steps, 500 for burnin)
res = SimpleMCMC.simpleNUTS(model, 1000)


