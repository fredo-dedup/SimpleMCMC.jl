load("SimpleMCMC.jl")
using SimpleMCMC

#  import SimpleMCMC.simpleRWM, SimpleMCMC.parseExpr, SimpleMCMC.expexp # windows

# linux
load("Distributions.jl")
using Distributions

# windows
load("distributions.jl") 
import Distributions.Gamma, Distributions.Normal, Distributions.Exponential
import Distributions.Poisson 
import Distributions.pdf, Distributions.logpdf


######### big lm  10.000 obs x 40 var  ###########
begin
	srand(1)
	n = 10000
	nbeta = 40
	X = [fill(1, (n,)) randn((n, nbeta-1))]
	beta0 = randn((nbeta,))
	Y = X * beta0 + randn((n,))
end

model = quote
	vars::vector(nbeta)

	vars ~ Normal(0, 1)
	resid = Y - X * vars
	resid ~ Normal(0, 1)
end

res = simpleRWM(model, 10000)

dlmwrite("/tmp/mjcl.txt", res)


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




