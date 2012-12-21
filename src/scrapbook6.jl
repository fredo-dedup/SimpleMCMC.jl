load("newlib.jl")

e = :(a=b+1)

e

processExpr(:(a= b+2), :unfold)
processExpr(:(for a in 1:3:a+=2;end), :unfold)

unfold_equal(:(a = b +2))



f = function("exp")
f
apply(Function::"exp",2.0)
typeof(f)

Function
