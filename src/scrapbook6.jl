load("newlib.jl")

import SimpleMCMC.expexp
using SimpleMCMC

SimpleMCMC.processExpr(:(a= b+2), :unfold)
SimpleMCMC.processExpr(:(a= b+2c), :unfold)
SimpleMCMC.processExpr(:(a[12] = sum(z[i:j])), :unfold)
SimpleMCMC.processExpr(:(for a in 1:3:a+=2;end), :unfold)

model = quote
	b::scalar
	k::vector(5)
	
	a = b+6
	x = sin(k * z)
end

expexp(model)
model2 = SimpleMCMC.processExpr(model, :unfold)
expexp(model2)

avars = SimpleMCMC.processExpr(model2, :listVars, Set{Symbol}(:b))
avars = SimpleMCMC.processExpr(model2, :listVars, Set{Symbol}(:k))
avars = SimpleMCMC.processExpr(model2, :listVars, Set{Symbol}())

SimpleMCMC.processExpr(model, :findParams, SimpleMCMC.Parmap(Dict{Symbol, Expr}(), 0))
x=1
test(x) = (x+=1)
