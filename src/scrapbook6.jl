load("newlib.jl")

include("../../Distributions.jl/src/distributions.jl")
using Distributions

import SimpleMCMC.expexp

SimpleMCMC.unfold(:(a= b+2))
SimpleMCMC.unfold(:(b+2))
SimpleMCMC.unfold(:(a= b+2c))
SimpleMCMC.unfold(:(a[12] = sum(z[i:j])))
SimpleMCMC.unfold(:(for a in 1:3:a+=2;end))

model = quote
	b::scalar
	k::vector(5)
	
	a = b+6
	x = sin(k * z)
	x ~ Weibull(2,a)
end

(f, n) = SimpleMCMC.buildFunction(model)
z = 12.0
eval(f)
__loglik([1.,1,1,1,1,1])

(f, n) = SimpleMCMC.buildFunctionWithGradient(model)
eval(f)
__loglik([1.,1,1,1,1,1])

(model2, nparams, pmap) = SimpleMCMC.findParams(model)
# model2 = SimpleMCMC.translateTilde(model2)
model2 = SimpleMCMC.translateTilde2(model2)
model2 = SimpleMCMC.unfold(model2)
avars = SimpleMCMC.listVars(model2, keys(pmap))
dmodel = SimpleMCMC.backwardSweep(model2, avars)

model = quote
	x::scalar
	x ~ Weibull(1, 2)
end
(model2, nparams, pmap) = SimpleMCMC.findParams(model)
model3 = SimpleMCMC.translateTilde(model2)
model4 = SimpleMCMC.translateTilde2(model2)
model4 = SimpleMCMC.unfold(model4)
avars = SimpleMCMC.listVars(model4, keys(pmap))
dmodel = SimpleMCMC.backwardSweep(model4, avars)
dmodel

x = 1.0
__acc = 0.0
eval(model21)
model21
eval(model4)

dset = Expr(:block, {:($(symbol("__d$v")) = zero($(symbol("$v")))) for v in avars}, Any)
eval(dset)
avars
d__acc = 0.0
d##tmp#19 = 0.0
, d##tmp#18, dx = 0.0, 0.0, 0.0, 0.0


model = quote
	k::vector(3)
	
	a = k[1]
	b = k[2:3]
	x = sum(b) + e^a
end

(model2, nparams, pmap) = SimpleMCMC.findParams(model)
model2 = SimpleMCMC.unfold(model2)
avars = SimpleMCMC.listVars(model2, keys(pmap))
dmodel = SimpleMCMC.backwardSweep(model2, avars)

model = quote
	x ~ Normal(k, sin(s))
end
model2 = SimpleMCMC.translateTilde2(model)
SimpleMCMC.unfold(model2)
expexp(model2)

model = quote 
	acc = +(acc, sum(logpdfNormal(5,2,x)) )
end
expexp(model)

SimpleMCMC.unfold(model)
