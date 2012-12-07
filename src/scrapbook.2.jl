############# smaller sets  ###########
load("SimpleMCMC.jl")
import SimpleMCMC.simpleRWM, SimpleMCMC.parseExpr

load("distributions.jl") 
import Distributions.Gamma, Distributions.Normal, Distributions.Exponential
import Distributions.Poisson 
import Distributions.pdf, Distributions.logpdf

counts = [4, 0.01 ,1, 2 ,5 ,6, 4 ,1 ,0.001 ,2 ,3]

model = quote
	freq::scalar

	freq ~ Gamma(2,1)
	counts ~ Normal(max(0,freq), 1)
end

res = simpleRWM(model, 1000)

dlmwrite("c:/temp/mjcl.txt", res)

