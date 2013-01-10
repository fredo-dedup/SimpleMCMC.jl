######### linear regression 1000 obs x 10 var  ###########

load("simple-mcmc/src/SimpleMCMC.jl")
using SimpleMCMC

require("Distributions.jl/src/distributions.jl")
using Distributions

# simulate dataset
begin
	srand(1)
	n = 1000
	nbeta = 10
	X = [fill(1, (n,)) randn((n, nbeta-1))]
	beta0 = randn((nbeta,))
	Y = X * beta0 + randn((n,))
end

# define model
model = quote
	vars::vector(nbeta)

	vars ~ Normal(0, 1.0)  # Normal prior, variance 1.0 for predictors
	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  
end

# (func, np) = SimpleMCMC.buildFunctionWithGradient(model)
# (func, np) = SimpleMCMC.buildFunction(model)
# eval(func)
# __loglik([0.9 for i in 1:11])

load("simple-mcmc/src/SimpleMCMC.jl"); 
res = SimpleMCMC.simpleRWM(model, 10, 2)


(a,b) = 
SimpleMCMC.simpleRWM(model, 200, 100)
f = n -> SimpleMCMC.simpleRWM(model, n, 10, 0.9)
f(1000)

simpleRWM

tmp[1,1]


size(res)
res
dlmwrite("/tmp/mjcl.txt", res)

__loglik([ 1. for i in 1:41])

############  small lm  ###############

begin
	srand(1)
	n = 100
	nbeta = 4
	X = [fill(1, (n,)) randn((n, nbeta-1))]
	beta0 = randn((nbeta,))
	Y = X * beta0 + randn((n,))
end

res = simpleRWM(model, 100000)

dlmwrite("/tmp/mjcl.txt", res)

############  binary logistic response  ###############

begin
	srand(1)
	n = 100
	nbeta = 4
	X = [fill(1, (n,)) randn((n, nbeta-1))]
	beta0 = randn((nbeta,))
	Y = (1/(1+exp(-(X * beta0)))) .> rand(n)
end
mean(Y)

model = quote
	vars::vector(nbeta)

	vars ~ Normal(0, 1)
	resid = Y - (1/(1+exp(- X * vars)))
	resid ~ Normal(0, 1)
end


res = simpleRWM(model, 100000)

dlmwrite("/tmp/mjcl.txt", res)




