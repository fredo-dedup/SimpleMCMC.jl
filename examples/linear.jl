######### linear regression 1000 obs x 10 var  ###########

using SimpleMCMC

# simulate dataset
srand(1)
n = 1000
nbeta = 10 # number of predictors, including intercept
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = X * beta0 + randn((n,))

# define model
model = quote
	beta ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	resid = Y - X * beta
	resid ~ Normal(0, 1.0)  
end

# run random walk metropolis (10000 steps, 5000 for burnin)
res = simpleRWM(model, steps=10000, burnin=1000, beta=zeros(nbeta))

res.acceptRate  # acceptance rate
[ mapslices(mean, res.params[:beta], 2) beta0 ] # show calculated and original coefs side by side

# run Hamiltonian Monte-Carlo (1000 steps, 500 for burnin, 2 inner steps, 0.05 inner step size)
res = simpleHMC(model, steps=1000, isteps=2, stepsize=0.05, beta=zeros(nbeta))

# run NUTS - HMC (1000 steps, 500 for burnin)
res = simpleNUTS(model, steps=1000, beta=zeros(nbeta))


