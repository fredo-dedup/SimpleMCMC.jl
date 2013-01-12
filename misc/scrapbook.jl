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





