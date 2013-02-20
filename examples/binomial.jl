######### logistic regression on 1000 obs x 10 var  ###########

include("../src/SimpleMCMC.jl")

# simulate dataset
srand(1)
n = 1000
nbeta = 10 # number of predictors, including intercept
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

# define model
model = quote
	vars::real(nbeta)

	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	resid = Y - ( 1 / (1. + exp(X * vars)) )
	resid ~ Normal(0, 1.0)  
end

# run random walk metropolis (10000 steps, 1000 for burnin)
res = SimpleMCMC.simpleRWM(model, 10000, 1000)

mean(res[:,2]) # accept rate
[ [mean(res[:,i+2])::Float64 for i in 1:nbeta] beta0 ] # calculated and original values side by side

# run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 3 inner steps, 0.2 inner step size)
res = SimpleMCMC.simpleHMC(model, 10000, 1000, 3, 2e-1)

mean(res[:,2]) # accept rate
[ [mean(res[:,i])::Float64 for i in 3:size(res,2)] beta0 ] # calculated and original values side by side

# run NUTS HMC (10000 steps, 1000 for burnin)
res = SimpleMCMC.simpleNUTS(model, 10000, 1000, zeros(nbeta))

mean(res[:,2]) # accept rate
[ [mean(res[:,i])::Float64 for i in 3:size(res,2)] beta0 ] # calculated and original values side by side


