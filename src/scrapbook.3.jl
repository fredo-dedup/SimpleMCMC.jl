############# big lm  10.000 obs x 40 var  ###########
load("SimpleMCMC.jl")
import SimpleMCMC.simpleRWM, SimpleMCMC.parseExpr

load("distributions.jl") 
import Distributions.Gamma, Distributions.Normal, Distributions.Exponential
import Distributions.Poisson 
import Distributions.pdf, Distributions.logpdf

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

res = simpleRWM(model, 1000)

dlmwrite("c:/temp/mjcl.txt", res)

