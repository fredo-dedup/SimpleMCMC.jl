######### hierarchical logistic regression  ###########
# using example of STAN 1.0.2 manual, chapter 10.7 p. 49

include("../src/SimpleMCMC.jl")

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

model = quote
	mu::real(D)
	sigma::real(D)
	beta::real(D,L)

	mu ~ Normal(0, 100)
	sigma ~ Uniform(0, 1000)

	beta ~ Normal(mu * onerow, (sigma.^2) * onerow)

	effect = (beta[ll,:] .* X) * onecol
	prob = 1. / ( 1. + exp(- effect) )
	Y ~ Bernoulli(prob)
end

res = SimpleMCMC.simpleRWM(model, 100000)
[Float64[mean(res[:,i+2]) for i in 1:(D+D+L*D)][9:28] reshape(beta0, 20)]

res = SimpleMCMC.simpleHMC(model, 10000, 1000, 1., 2, 0.001)

res = SimpleMCMC.simpleNUTS(model, 10)


my, np = SimpleMCMC.buildFunctionWithGradient(model)
eval(my)

ll, grad = __loglik(ones(D+D+L*D))
ll, grad = __loglik(ones(D+D+L*D))

ll
grad

