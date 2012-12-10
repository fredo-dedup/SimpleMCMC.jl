############# smaller sets  ###########

load("SimpleMCMC.jl")
# linux
using SimpleMCMC

# windows
import SimpleMCMC.simpleRWM, SimpleMCMC.parseExpr

# windows
load("distributions.jl") 
import Distributions.Gamma, Distributions.Normal, Distributions.Exponential
import Distributions.Poisson 
import Distributions.pdf, Distributions.logpdf

## linux
load("Distributions.jl")
using Distributions

counts = [4, 0.01 ,1, 2 ,5 ,6, 4 ,1 ,0.001 ,2 ,3]

model = quote
	freq::scalar

	freq ~ Gamma(2,1)
	counts ~ Normal(max(0,freq), 1)
end

res = simpleRWM(model, 1000)

min(res[:,1])

dlmwrite("c:/temp/mjcl.txt", res)

main.__loglik


for i in 1:1000
	println(logpdf(Gamma(2,1), 10*randn(1)))
end

##################################################################"

load("ADlib.jl")
using ADlib

beta = [1]
freq= ADVar(beta[1], 1, 1)
freq
freq + freq

lp = 0.0
lp += logpdf(Gamma(2,1),1)

logpdf(Gamma(2,1), freq)

	freq ~ Gamma(2,1)
	counts ~ Normal(max(0,freq), 1)
end

Gamma(2,1)

function logpdf(d::Gamma, x::ADVar)
    res = ADVar([0.0]
    for i in size(x,2) * size(x,3)
    x <= 0. ? -Inf : (-x/d.scale) - log(d.scale)
	end
end

x = ADVar([1])

function logpdf(d::Normal, x::ADVar)
    d.mean
    d.std

    tmp = zeros(nderiv(x)+1)

    res = ADVar([0.0]
    for i in size(x,2) * size(x,3)
    x <= 0. ? -Inf : (-x/d.scale) - log(d.scale)
	end
end




type Gamma <: ContinuousDistribution
    shape::Float64
    scale::Float64
    Gamma(sh,sc) = sh > 0 && sc > 0 ? new(float64(sh), float64(sc)) : error("Both schape and scale must be positive")
end

