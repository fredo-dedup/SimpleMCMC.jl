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

SimpleMCMC.listVars(model, [:b])

expexp(model)
model2 = SimpleMCMC.unfold(model)
SimpleMCMC.processExpr(model, :unfold)
expexp(model2)

avars = SimpleMCMC.processExpr(model2, :listVars, Set{Symbol}(:b))
avars = SimpleMCMC.processExpr(model2, :listVars, Set{Symbol}(:k))
avars = SimpleMCMC.processExpr(model2, :listVars, Set{Symbol}())

SimpleMCMC.processExpr(model, :findParams, SimpleMCMC.Parmap(Dict{Symbol, Expr}(), 0))
x=1
test(x) = (x+=1)

function x()
	global a
	a = 12
	y()
end
function y()
	b = a + 2
end
a=13

x()
y()
x() = (global a2; a2=1; y(); println(a2))
y() = (global a2; a2=a2+2)
a
y()
a2
function foo(n)
  x = 0
  for i = 1:n
    x = x + 1
  end
  x
end


foo(10)