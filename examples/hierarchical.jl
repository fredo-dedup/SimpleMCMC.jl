######### hierarchical logistic regression  ###########
# using example of STAN 1.0.2 manual, chapter 10.7 p. 49

using SimpleMCMC

## generate data set
N = 50 # number of observations
D = 4  # number of groups
L = 5  # number of predictors

srand(1)
mu0 = randn(1,L)
sigma0 = rand(1,L)
# model matrix nb group rows x nb predictors columns
beta0 = Float64[randn()*sigma0[j]+mu0[j] for i in 1:D, j in 1:L] 

oneD = ones(D)
oneL = ones(L)

ll = rand(1:D, N)  # mapping obs -> group
X = randn(N, L)  # predictors
Y = [rand(N) .< ( 1 ./ (1. + exp(- (beta0[ll,:] .* X) * oneL )))]

## define model
model = quote
	mu ~ Normal(0, 1)
	sigma ~ Weibull(2, 1)

	beta ~ Normal(oneD * mu, oneD * sigma)

	effect = (beta[ll,:] .* X) * oneL
	prob = 1. / ( 1. + exp(- effect) )
	Y ~ Bernoulli(prob)
end

# intial values for parameters
init = {:mu=> zeros(1,L),
		:sigma=> ones(1,L),
		:beta=> zeros(D,L)}

# run random walk metropolis (10000 steps, 1000 for burnin)
res = simpleRWM(model, steps=10000, burnin=1000; init...)

mapslices(mean, res.params[:beta], 3)
mapslices(mean, res.params[:sigma], 3) ; sigma0
mapslices(mean, res.params[:mu], 3) ; mu0

# run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 10 inner steps, 0.03 inner step size)
res = simpleHMC(model, steps=10000, burnin=1000, isteps=5, stepsize=0.03; init...)

# # run NUTS - HMC (1000 steps, 500 for burnin)
res = simpleNUTS(model, steps=10, burnin=1; init...)  # very slow  (bug ?)

res.misc[:jmax]  # number of splittings at each step
res.misc[:epsilon] # size of inner steps

