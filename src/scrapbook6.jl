load("newlib.jl")

import SimpleMCMC.expexp
#using SimpleMCMC

SimpleMCMC.unfold(:(a= b+2))
SimpleMCMC.unfold(:(a= b+2c))
SimpleMCMC.unfold(:(a[12] = sum(z[i:j])))
SimpleMCMC.unfold(:(for a in 1:3:a+=2;end))

model = quote
	b::scalar
	k::vector(5)
	
	a = b+6
	x = sin(k * z)
end

(model2, nparams, pmap) = SimpleMCMC.findParams(model)
model2 = SimpleMCMC.unfold(model2)
SimpleMCMC.listVars(model, keys(pmap))

expexp(model)
model2 = SimpleMCMC.unfold(model)
SimpleMCMC.processExpr(model, :unfold)
expexp(model2)

