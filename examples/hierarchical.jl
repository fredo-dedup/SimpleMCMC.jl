######### hierarchical logistic regression  ###########
# using example of STAN 1.0.2 manual, chapter 10.7 p. 49

include("../src/SimpleMCMC.jl")

## generate random values
N = 50 # number of observations
D = 4  # number of groups
L = 5  # number of predictors

onecol = ones(L, 1)
onerow = ones(1, L)

srand(1)
ll = rand(1:D, N)  # mapping obs -> group
X = randn(N, L)  # predictors
beta0 = randn(D, L)  # model matrix nb group rows x nb predictors columns
Y = [rand(N) .< ( 1 ./ (1. + exp(- (beta0[ll,:] .* X) * onecol)))]

## define model
model = quote
	mu::real(D)
	sigma::real(D)
	beta::real(D,L)

	mu ~ Normal(0, 100)
	sigma ~ Uniform(0, 1)

	beta ~ Normal(mu * onerow, (sigma.^2) * onerow)

	effect = (beta[ll,:] .* X) * onecol
	prob = 1. / ( 1. + exp(- effect) )
	Y ~ Bernoulli(prob)
end

# run random walk metropolis (10000 steps, 5000 for burnin)
res = SimpleMCMC.simpleRWM(model, 10000)

sum(res.params[:mu],2) / res.samples  # mu samples mean
sum(res.params[:sigma],2) / res.samples # sigma samples mean
sum(res.params[:beta],3) / res.samples # beta samples mean

# run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 2 inner steps, 0.1 inner step size)
res = SimpleMCMC.simpleHMC(model, 10000, 1000, 1., 2, 0.05)

# run NUTS - HMC (1000 steps, 500 for burnin)
res = SimpleMCMC.simpleNUTS(model, 10000)


