require("pkg")
Pkg.init()
Pkg.add("Distributions")
Pkg.rm("DataFrames")
Pkg.add("DataFrames")
Pkg.cd_pkgdir()  do  
    Pkg._resolve()
end

################################

include("../src/SimpleMCMC.jl")

function recap(res)
    print("accept : $(round(100*mean(res[:,2]),1))%, ")
    print("mean : $(round(mean(res[:,3]),3)), ")
    println("std : $(round(std(res[:,3]),3))")
end

model = :(x::real ; x ~ Weibull(1, 1))  # mean 1.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 3.400 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [1.], 1, 0.001)) # 9.500 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.]))  # 400-500 ess/s, mean is off by -0.05

model = :(x::real ; x ~ Weibull(3, 1)) # mean 0.89, std 0.325
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 5.900 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.6], 1, 0.01)) # 30.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.]))  # 15.000 ess/s, correct

model = :(x::real ; x ~ Uniform(0, 2)) # mean 1.0, std 0.577
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 5.000 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [1.], 2, 0.7)) # 14.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 10000, 1000, [1.]))  # 130 ess/s, very slow due to gradient == 0 ?

model = :(x::real ; x ~ Normal(0, 1)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.]))  # 6.000 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.], 2, 0.8)) # 45.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.2]))  # 20.000 ess/s, correct

model = :(x::real ; x ~ Normal(3, 12)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.]))  # 6.200 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.], 2, 9.)) # 45.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.]))  # 17.000 ess/s, correct

model = :(x::real(10) ; x ~ Normal(3, 12)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, 0.))  # 300 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, 0., 2, 0.8)) # 45.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, 0.))  # 4.800 ess/s, correct, (4.600 without direct call to libRmath)


z = randn(10) .< 0.5
model = :(x::real ; z ~ Bernoulli(x)) # mean 0.5, std ...
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.5]))  # 6.200 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.5], 2, 0.04)) # 140 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.5]))  # 17.000 ess/s, correct




model = :(x::real ; x ~ Weibull(1, 1)) 
recap(SimpleMCMC.simpleNUTS(model, 10000, 0, [0.2]))  
model = :(x::real ; x ~ Normal(0, 1)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleNUTS(model, 10000, 0, [0.2]))  

res = SimpleMCMC.simpleNUTS(model, 10000, 0, [0.2])
mean(res[:,3])

dlmwrite("c:/temp/dump.txt", res[:,3])

################################



dat = dlmread("c:/temp/ts.txt")
size(dat)

tx = dat[:,2]
dt = dat[2:end,1] - dat[1:end-1,1]

model = quote
    mu::real
    tau::real
    sigma::real

    tau ~ Weibull(2,1)
    sigma ~ Weibull(2, 0.01)
    mu ~ Uniform(0,1)

    f2 = exp(-dt / tau / 1000)
    resid = tx[2:end] - tx[1:end-1] .* f2 - mu * (1.-f2)
    resid ~ Normal(0, sigma^2)
end

res = SimpleMCMC.simpleRWM(model, 10000, 1000, [0.5, 30, 0.01])
res = SimpleMCMC.simpleRWM(model, 101000, 1000)
mean(res[:,2])

res = SimpleMCMC.simpleHMC(model, 10000, 500, [0.5, 0.01, 0.01], 2, 0.001)
res = SimpleMCMC.simpleHMC(model, 1000, 1, 0.1)
mean(res[:,2])

dlmwrite("c:/temp/dump.txt", res)
mean(Weibull(2,1))

###
model = quote
    mu::real
    scale::real

    scale ~ Weibull(2,1)
    mu ~ Uniform(0,1)

    f2 = exp(-dt * scale)
    resid = tx[2:end] - tx[1:end-1] .* f2 - mu * (1.-f2)
    resid ~ Normal(0, 0.01)
end

res = SimpleMCMC.simpleRWM(model, 10000, 500, [0.5, 0.01])
res = SimpleMCMC.simpleRWM(model, 101000, 1000)
mean(res[:,2])

res = SimpleMCMC.simpleHMC(model, 10000, 500, [0.5, 0.01], 5, 0.001)
res = SimpleMCMC.simpleHMC(model, 1000, 1, 0.1)
mean(res[:,2])

dlmwrite("c:/temp/dump.txt", res)

#####################################################################

require("DataFrames")
using DataFrames

BitArray


type MCMCRun2
    acceptRate::Float64
    time::Float64
    ess::Range{Int32}
    essBySec::Range{Float64}
    params::Dict
end

x = MCMCRun2(.25, 5.6, 4:1000, 4.2:10.3, {:vars=>[1 2 3 ; 4 5 6 ]})



##########  creates a parameterized type to ease AST exploration  ############
type ExprH{H}
    head::Symbol
    args::Vector
    typ::Any
end
toExprH(ex::Expr) = ExprH{ex.head}(ex.head, ex.args, ex.typ)
toExpr(ex::ExprH) = expr(ex.head, ex.args)

typealias Exprequal    ExprH{:(=)}
typealias Exprdcolon   ExprH{:(::)}
typealias Exprpequal   ExprH{:(+=)}
typealias Exprcall     ExprH{:call}
typealias Exprblock    ExprH{:block}
typealias Exprline     ExprH{:line}
typealias Exprref      ExprH{:ref}
typealias Exprif       ExprH{:if}

##########  helper function to get symbols appearing in AST ############
getSymbols(ex::Expr) =       getSymbols(toExprH(ex))
getSymbols(ex::ExprH) =      Set{Symbol}()
getSymbols(ex::Symbol) =     Set{Symbol}(ex)
getSymbols(ex::Exprref) =    Set{Symbol}(ex.args[1])
getSymbols(ex::Exprequal) =  union(getSymbols(ex.args[1]), getSymbols(ex.args[2]))
getSymbols(ex::Exprcall) =   mapreduce(getSymbols, union, ex.args[2:end])
getSymbols(ex::Exprif) =     mapreduce(getSymbols, union, ex.args)
getSymbols(ex::Exprblock) =  mapreduce(getSymbols, union, ex.args)

ex = quote
    x::real

    a = b[3]
    c = 12a
    z[3] = "abcd"
    c = exp(max(a,b)) / 3.0
end

dump(ex)

getSymbols(ex)
f, n, pmap = SimpleMCMC.parseModel(ex, true)
pmap[:x]
