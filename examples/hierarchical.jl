######### hierarchical logistic regression  ###########
# using example of STAN 1.0.2 manual, chapter 10.7 p. 49

	for (n in 1:N) y[n] ~ bernoulli(inv_logit(x[n] * beta[ll[n]]));


N = 50 # number of observations
D = 4  # number of groups
L = 5  # number of predictors

ll = int(ceil(rand(N)*D))  # mapping obs -> group
X = randn(N, L)  # predictors
beta0 = randn(D, L)  # model matrix nb group rows x nb predictors columns
Y = rand(N) .< ( 1 ./ (1. + exp(- (beta0[ll,:] .* X) * onecol)))

onecol = ones(L, 1)
onerow = ones(1, L)

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

res = SimpleMCMC.simpleRWM(model, 1000)

[ [mean(res[:,i+2])::Float64 for i in 1:nbeta] beta0 ] 

SimpleMCMC.simpleNUTS(model, 10)


reshape(x, (5,50))