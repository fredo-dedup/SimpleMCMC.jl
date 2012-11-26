load("extras/distributions.jl")
import Distributions.*

load("p:/documents/julia/mc_lib.2.jl")

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

		modelExpr = mcmc2(model)
		setmap = :(sigma=beta[1] ; vars=beta[2:(nbeta+1)])
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

ex = params
function parseParams(ex)
	println(ex.head, " -> ")
	@assert params.head == :block
	index1 = 1
	index2 = 1
	assigns = {}
	for e in ex.args
		if e.head == :(::)
			@assert typeof(e.args[1]) == Symbol
			@assert typeof(e.args[2]) == Expr
			e2 = e.args[2]
			if e2.args[1] == :scalar
				push(assigns, :($(e.args[1]) = beta[$index1]))
				index1 += 1
			elseif typeof(e2.args[1]) == Expr
				e3 = e2.args[1].args
				if e3[1] == :vector
					nb = eval(e3[2])
					push(assigns, :($(e.args[1]) = beta[$index1:$(nb+index1-1)]))
					index1 += nb
				end
			end
		end
	end

	Expr(:block, assigns, Any)
end

e = params.args[2]
e.head


my = :()
expexp(my)
expexp(params)
my.args[1] = :(a=2)


expexp(params)


###################  runs  ##########################
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