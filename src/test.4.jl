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

	begin
		loop = quote 
		 	for __i in 1:steps
				oldbeta, beta = beta, beta + randn(1+nbeta) * scale

				$setmap
				# eval(setmap)
		 		old__lp, __lp = __lp, 0.0

				$modelExpr
				# eval(modelExpr)
				if rand() > exp(__lp - old__lp)
					__lp, beta = old__lp, oldbeta
				end
				samples[__i, :] = vcat(__lp, beta)
			end
		end
	end

params = quote
	sigma::{scalar, init=0, support=(0, Inf)}
	vars::{vector(nbeta), init=0}
end

parmap = parseParams(params)

my = :()
expexp(my)
expexp(params)
my.args[1] = :(a=2)

expexp(params)


###################  runs  ##########################
expexp(:(4+5*a))
expexp(:(sigma::{scalar}, vars::{vector(nbeta)}))

params = :(sigma::{scalar}, vars::{vector(nbeta)})

load("SimpleMCMC.jl")

eval(quote 
    sigma = beta[1]
    vars = beta[2:41]
end)

beta = ones(41)
params = :(sigma::{scalar}, vars::{vector(nbeta)})

nbeta = 40
simpleMCMC10(model, :(sigma::{scalar}, vars::{vector(nbeta)}), 10)

@time begin
	samples = simpleMCMC10(model, :(sigma::{scalar}, vars::{vector(nbeta)}), 1000)
end  # 9,6 pour 1000



test = quote 
	lp = 1.2
	println("lp :", lp)
end

(nbeta, parmap) = parseParams(:(sigma::{scalar}, vars::{vector(nbeta)}))
model2 = parseModel(model)

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












	begin
		scale = 0.1
		steps = 10000
		samples = zeros(Float64, (steps, 2+nbeta))
	end

	modelExpr

	params = :( sigma, vars(n) [0,10], other [1,Inf])
	expexp(params)

model

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

	@time begin
		eval(loop)
	end    # 1.05 pour 10000 iter, un peu plus lent, 73s pour 40 vars et 10.000 iter

	mean(samples[:,5])  # mais la distrib est meilleure (mes fonctions sont fausses !)
	[[mean(samples[:,i]) for i in 1:(nbeta+1)] ; beta0 ]
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