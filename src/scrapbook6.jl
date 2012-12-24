load("newlib.jl")

import SimpleMCMC.processExpr
using SimpleMCMC

SimpleMCMC.processExpr(:(a= b+2), :unfold)
SimpleMCMC.processExpr(:(a= b+2c), :unfold)
SimpleMCMC.processExpr(:(a[12] = sum(z[i:j])), :unfold)
SimpleMCMC.processExpr(:(for a in 1:3:a+=2;end), :unfold)

model = quote
	a = b+6
	x = sin(k * z)
end

expexp(model)
SimpleMCMC.processExpr(model, :unfold)





