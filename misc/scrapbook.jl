load("pkg")
Pkg.init()
Pkg.add("Distributions")
Pkg.update("Distributions")
Pkg.add("DataFrames")

require("../../.julia/Distributions.jl/src/Distributions.jl")
using Distributions
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
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.], 2, 0.8)) # 45.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.]))  # 17.000 ess/s, correct






model = :(x::real ; x ~ Weibull(1, 1)) 
recap(SimpleMCMC.simpleNUTS(model, 10000, 0, [0.2]))  
model = :(x::real ; x ~ Normal(0, 1)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleNUTS(model, 10000, 0, [0.2]))  

res = SimpleMCMC.simpleNUTS(model, 10000, 0, [0.2])
mean(res[:,3])

0ÓK], -0.9389385332046728, -0.2 
 4.0 
 0.7, 1000, 0.05, 0.75, 10 
 0.0, 3.6888794541139363, 0.0

p!ç], -0.9389385332046728, -0.2 
 4.0 
 0.7, 1000, 0.05, 0.75, 10 
 0.0, 3.6888794541139363, 0.0

__loglik([0.2])[1]
__loglik([1.])[1]
__loglik([-1.])[1]
__loglik([0.636822])[1]
__loglik([0.636822])[2]
__loglik([-0.636822])[1]
__loglik([-0.636822])[2]

res = SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.])
dlmwrite("c:/temp/dump.txt", res[:,3])

########################################################################
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


model = quote
    x::real
    x ~ Normal(0,1)
end

res = SimpleMCMC.simpleRWM(model, 10000)
res2 = {:llik=>[1,2,3,3,4], :accept=>[0,1,0,1,0,1,1,1,0,0,0],
:x=>[4,4,4,5,5,5,6,6,3,22,3]}
mean(res2[:accept])
dump(res2)

res = SimpleMCMC.simpleHMC(model, 10000, 5000, 3, 0.5)
[ (mean(res[:,i]), std(res[:,i])) for i in 3:size(res,2)]
=======
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




###########################################################

f() = 2.0
g(f) = f()
g(f)
# 2.0

f() = 3.0
g(f)
# 2.0

