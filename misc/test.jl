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

res = SimpleMCMC.simpleNUTS(model, 1000)

