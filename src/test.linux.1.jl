### pour linux
load("pkg")      # this will go away soon
Pkg.add("Distributions") # installs MyPkg and dependencies
require("Distributions.jl") # will also make this go away
using Distributions      # this will trigger require
import Distributions.Normal, Distributions.Gamma, Distributions.logpdf

load("~/.julia/Distributions/src/Distributions.jl")
mean(Normal(1,1))

### pour windows
load("extras/distributions.jl")
import Distributions.*

# load("mc_lib.3.jl")
load("SimpleMCMC.jl")
# include("SimpleMCMC.jl")
# import SimpleMCMC.*

simpleMCMC
###################  problem data  ##########################
	begin
		srand(1)
		n = 10000
		nbeta = 40
		X = [fill(1, (n,)) randn((n, nbeta-1))]
		beta0 = randn((nbeta,))
		Y = X * beta0 + randn((n,))

		model = quote
			resid = Y - X * vars
			sigma ~ Gamma(2, 1)

			vars ~ Normal(0.0, 1)
			resid ~ Normal(0.0, sigma)
		end

		# modelExpr = mcmc2(model)
		# setmap = :(sigma=beta[1] ; vars=beta[2:(nbeta+1)])
	end

###################  runs  ##########################
load("SimpleMCMC.jl")  # refresh if necessary


simpleMCMC10(model, :(sigma::{scalar}, vars::{vector(nbeta)}), 100)

@time begin
	samples = simpleMCMC10(model, :(sigma::{scalar}, vars::{vector(nbeta)}), 10000, 5000)
end  
# 9,6 pour 1000
# 15.2 pour 10000-5000 (linux)

	[[mean(samples[:,i])::Float64 for i in 3:(nbeta+3)] [1, beta0] ]


dlmwrite("c:/temp/mcjl.txt", samples, '\t')


(nbeta, parmap) = parseParams(:(sigma::{scalar}, vars::{vector(nbeta)}))
model2 = parseModel(model)


###################  sandbox   #########################
