
load("extras/distributions.jl")
import Distributions.*

load("p:/documents/julia/mc_lib.2.jl")
#load("p:/documents/julia/Distributions.jl/src/Distributions.jl")
#load("p:/documents/julia/Distributions.jl/src/Distribs.jl")

###################  runs  ##########################
	scale = 0.1
	steps = 10000
	samples = zeros(Float64, (steps, 2+nbeta))
	modelExpr

#############  find starting values  ################

	for k in 1:10
		global beta
		global __lp
		beta = randn(nbeta+1) .* scale
		eval(setmap)
		__lp = 0.0
		eval(modelExpr)
		if __lp > -Inf 
			break 
		end
	end

beta = [1 , zeros((40,)) ]

quote 	#  line 2:
    resid = -(Y, *(X, vars))	#  line 3:
    __lp = +(__lp, sum(logpdf(Gamma(2, 1), sigma)))	#  line 5:
    __lp = +(__lp, sum(logpdf(Normal(0.0, 1), vars)))	#  line 6:
    __lp = +(__lp, sum(logpdf(Normal(0.0, sigma), resid)))
end


	@time begin
		eval(loop)
	end    # 1.05 pour 10000 iter, un peu plus lent, 73s pour 40 vars et 10.000 iter

	mean(samples[:,5])  # mais la distrib est meilleure (mes fonctions sont fausses !)
	[mean(samples[:,i]) for i in 1:(nbeta+1)]
	 #-142.638   
	 #  -0.946342
	 #  -0.978874
	 #  -0.977102
	 #  -1.7795  
	beta0


dlmwrite("c:/temp/mcjl.txt", samples, '\t')

##########################################################

x = rand(Normal(0.0, 1.0), 10000)
size(x)
x[1:20]

d = Gamma(2, 0.5)
pdf(d, 0.2)
[ pdf(d, x/10) for x in 1:10]
pdf(d, [0:100] / 10.0)
logpdf(Normal(0.0, 1.0), [0.0, 1.2, 3.0])

d
0.1:2.1:0.1