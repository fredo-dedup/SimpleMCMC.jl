 load("pkg")

 Pkg.init()

Pkg.add("Distributions")
Pkg.update("Distributions")


whos()


model = quote
	x::scalar

	resid = (x-2)*(x-1)
	resid ~ Normal(0, 0.01)  
end

myf, np = SimpleMCMC.buildFunctionWithGradient(model)
np
myf

eval(myf)

l0, grad = __loglik([1.])
l0, grad = __loglik([1.01])
l0, grad = __loglik([2.])
l0, grad = __loglik([2.01])
l0, grad = __loglik([1.99])
l0, grad = __loglik([1.01])

################################
Y = [1., 2, 3, 4]
X = [0. 1; 0 1; 1 1; 1 2]

model = quote
	vars::vector(2)

	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  
end

myf, np = SimpleMCMC.buildFunctionWithGradient(model)
eval(myf)

res = SimpleMCMC.simpleHMC(model,100, 5, 1.5, 10, 0.1)

l0, grad = __loglik(ones(2))

[ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]


vars = ones(2)
resid=Y - X * vars
dresid = zeros(4)
dresid += /(sum(+(-(resid),0)),1.0)
dvars = zeros(2) 
dvars += transpose(X) * dresid


################################

myf, np = SimpleMCMC.buildFunctionWithGradient(model)
eval(myf)

l0, grad = __loglik(ones(nbeta))
[ [ (beta=ones(nbeta) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:nbeta ] grad]



################################

model = quote
	x::real
	x ~ Normal(0., 1.)  
end

include("../src/SimpleMCMC.jl")
require("Distributions")
using Distributions
import Distributions.Uniform


myf, np = SimpleMCMC.buildFunction(model)
eval(myf)
myf, np = SimpleMCMC.buildFunctionWithGradient(model)
eval(myf)

SimpleMCMC.expexp(:( a += b))

__loglik([1.])
__loglik([1.01])

res = SimpleMCMC.simpleRWM(model,10000)

writedlm("/tmp/dump.txt", res)

################################
model = quote
	b::real
	k::real(5)
	
	a = b+6
	x = sin(dot(k, z))

	x ~ Weibull(a, 2.0)
end