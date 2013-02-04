load("pkg")
Pkg.init()
Pkg.add("Distributions")
Pkg.update("Distributions")
Pkg.add("DataFrames")

################################
Y = [1., 2, 3, 4]
X = [0. 1; 0 1; 1 1; 1 2]

model = quote
	x::real
	x ~ Normal(0, 1.0)  
end

require("../src/SimpleMCMC.jl")
include("../src/SimpleMCMC.jl")
require("../../.julia/Distributions.jl/src/Distributions.jl")

res = SimpleMCMC.simpleRWM(model, 100000)
mean(res[:,2])
[mean(res[:,3]) std(res[:,3])]

res = SimpleMCMC.simpleHMC(model, 100000, 2, 1.0)
mean(res[:,2])
[mean(res[:,3]) std(res[:,3])]

dlmwrite("c:/temp/dump.txt", res)


res = SimpleMCMC.simpleNUTS(model, 100)
mean(res[:,2])
[mean(res[:,3]) std(res[:,3])]



myf, np = SimpleMCMC.buildFunctionWithGradient(model)
eval(myf)

res = SimpleMCMC.simpleHMC(model,100, 5, 1.5, 10, 0.1)

l0, grad = __loglik(ones(2))
[ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]


################################################

Y = [1,2,3]
model = :(x::real(3); y=Y; y[2] = x[1] ; y ~ TestDiff())

## ERROR : marche pas, notamment parce que la subst des noms de variable ne fonctionne pas


    myf, np = SimpleMCMC.buildFunctionWithGradient(model)
    ex2 = expr(:block, myf.args[2].args)
    __beta = [3., 10, 0]
    a, b =eval(ex2)
    __beta = [3., 11, 0]
    c, d = eval(ex2)

    cex = expr(:block, vcat({:(__beta = [1.0])}, ex2))


#####################################################

type Hexpr{T}
    head::Symbol
    args::Vector
    typ::Any
end

toHexpr(ex::Expr) = Hexpr{ex.head}(ex.head, ex.args, ex.typ)
toExpr(ex::Hexpr) = expr(ex.head, ex.args)

test(ex::Hexpr{:block}) =   println("block : $ex")
test(ex::ExprBlock) =   println("block : $ex")
test(ex::Hexpr{:line}) =    println("line : $ex")
test(ex::ExprRef) =    println("ref : $ex")
test(ex::Hexpr{:call}) =    println("call : $ex")
test(ex::Hexpr{:(=)}) =     println("= : $ex")
test(ex::Hexpr) =           println("whatever ($(ex.head)) : $ex")

test(ex::Hexpr{:(=)}) =     println("= : $ex")

typealias ExprBlock Hexpr{:block}
typealias ExprRef Hexpr{:ref}

test(ex::Expr) = test(toHexpr(ex))

a = :(a=b+5)
b = :(sin(f))
c = :((a=f;f+6))
d = :(f==3 ? x : y)
test(:(f[3]))

test(a)
test(b)
test(c)
test(d)


################################

