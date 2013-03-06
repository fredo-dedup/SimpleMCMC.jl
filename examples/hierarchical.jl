######### hierarchical logistic regression  ###########
# using example of STAN 1.0.2 manual, chapter 10.7 p. 49

include("../src/SimpleMCMC.jl")

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
	mu::real(1,L)
	sigma::real(1,L)
	beta::real(D,L)

	mu ~ Normal(0, 1)
	sigma ~ Uniform(0, 5)

	# beta ~ Normal(mu * onerow, (sigma.^2) * onerow)
	beta ~ Normal(oneD * mu, oneD * sigma)

	effect = (beta[ll,:] .* X) * oneL
	prob = 1. / ( 1. + exp(- effect) )
	Y ~ Bernoulli(prob)
end

# run random walk metropolis (10000 steps, 1000 for burnin)
res = SimpleMCMC.simpleRWM(model, 100000, 1000)

sum(res.params[:mu],3) / res.samples  # mu samples mean
sum(res.params[:sigma],3) / res.samples # sigma samples mean
sum(res.params[:beta],3) / res.samples # beta samples mean

# # run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 4 inner steps, 0.1 inner step size)
res = SimpleMCMC.simpleHMC(model, 10000, 1000, 10, 0.03)

# # run NUTS - HMC (1000 steps, 500 for burnin)
res = SimpleMCMC.simpleNUTS(model, 1000)  # very slow  (bug ?)
