######### logistic regression on binary response  ###########
using SimpleMCMC

# simulate dataset
srand(1)
n = 1000
nbeta = 10 # number of predictors, including intercept
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

# define model
model = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

# run random walk metropolis (10000 steps, 1000 for burnin)
res = simpleRWM(model, steps=10000, burnin=1000, vars=zeros(nbeta))

mean(res.loglik)
[ sum(res.params[:vars],2) / res.samples beta0] # calculated vs original coefs

# run Hamiltonian Monte-Carlo (1000 steps, 100 for burnin, 2 inner steps, 0.1 inner step size)
res = simpleHMC(model, isteps=2, stepsize=0.1, vars=zeros(nbeta))

# run NUTS HMC (10000 steps, 1000 for burnin)
res = simpleNUTS(model, steps=10000, burnin=1000, vars=zeros(nbeta))
