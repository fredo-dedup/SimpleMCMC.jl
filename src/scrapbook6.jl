load("newlib.jl")
# include("newlib.jl")

import SimpleMCMC.expexp
using SimpleMCMC

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

(model2, nparams, pmap) = SimpleMCMC.findParams(model)
# model2 = SimpleMCMC.translateTilde(model2)
model2 = SimpleMCMC.translateTilde2(model2)
model2 = SimpleMCMC.unfold(model2)
avars = SimpleMCMC.listVars(model2, keys(pmap))
dmodel = SimpleMCMC.backwardSweep(model2, avars)

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

