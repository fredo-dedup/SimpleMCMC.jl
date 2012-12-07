
############# biglm (10000 obs * 40 variables)  ###########
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

load("SimpleMCMC.jl")
import SimpleMCMC.simpleRWM

load("distributions.jl") 
using Distributions  # linux
import Distributions.Gamma, Distributions.Normal, Distributions.Exponential
import Distributions.Poisson 
import Distributions.pdf, Distributions.logpdf

res = simpleRWM(model, 1000)
res = simpleRWM(model, 100)
res = simpleRWM(model, 100, 10, 1.0)
res = simpleRWM(model, 100, 10, [1.0 for i in 1:40])



############# smaller sets  ###########
load("SimpleMCMC.jl")
import SimpleMCMC.simpleRWM, SimpleMCMC.parseExpr

load("distributions.jl") 
using Distributions  # linux
import Distributions.Gamma, Distributions.Normal, Distributions.Exponential
import Distributions.Poisson 
import Distributions.pdf, Distributions.logpdf

srand(1)
counts = [4, 0.01 ,1, 2 ,5 ,6, 4 ,1 ,0.001 ,2 ,3]

logpdf(Normal(0,1), [0.1 for i in 1:10000])

model = quote
	freq::scalar

	freq ~ Gamma(2,1)
	counts ~ Normal(freq, 1)
end

res = simpleRWM(model, 300)

logpdf(Exponential(1.0), 0.)

draws = rand(Poisson(4),10)
pdf(Poisson)
