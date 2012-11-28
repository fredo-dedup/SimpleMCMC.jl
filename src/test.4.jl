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
	samples = simpleMCMC10(model, :(sigma::{scalar}, vars::{vector(nbeta)}), 1000)
end  # 9,6 pour 1000

	[[mean(samples[:,i]) for i in 1:(nbeta+1)] ; beta0 ]
	 #-142.638   
	 #  -0.946342
	 #  -0.978874
	 #  -0.977102
	 #  -1.7795  
	beta0

dlmwrite("c:/temp/mcjl.txt", samples, '\t')


(nbeta, parmap) = parseParams(:(sigma::{scalar}, vars::{vector(nbeta)}))
model2 = parseModel(model)


###################  sandbox   #########################
	eval(quote 
		function f(beta) 
			local __lp

			$parmap
			__lp = 0.0 
			$model2
			__lp
		end
	end)

	f()

	test = :(sigma = alpha)
	f(x) = (zeta = x)
	eval(:(f() = (sigma = alpha)))
	f
	global sigma
	sigma
	f(5)
	zeta
	sigma

	function test2()
		local zeta
		# zeta = 0
		alpha = 3

		eval(:(f(x) = (:zeta = x)))

		f(alpha)
		println(zeta)
	# catch x
	# 	println("error : ", x)
	end

	test2()

	alpha
	sigma

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

##########################################################

	test = randn(4,4)
	test = test * test'
	det(test)
	res = chol(test)
	res' * res
	res

