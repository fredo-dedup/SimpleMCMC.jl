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
using Distributions

res = SimpleMCMC.simpleRWM(model, 100000)
mean(res[:,2])
[mean(res[:,3]) std(res[:,3])]

res = SimpleMCMC.simpleHMC(model, 100000, 500, [0.0], 1, 0.1)
mean(res[:,2])
[mean(res[:,3]) std(res[:,3])]

dlmwrite("c:/temp/dump.txt", res)

res = SimpleMCMC.simpleNUTS(model, 100000, 500, [0.0])
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

################################

dump(:(b = x==0 ? x : y))
dump(:(x==0 ))
dump(:(x!=0 ))
dump(:(x!=0 ))


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
