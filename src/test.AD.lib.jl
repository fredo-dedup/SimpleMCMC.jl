
load("AD.lib.jl")


ADVar(1, [2, 3]) + 5.0

########## scrapbook  #######################
c = 3
test(x) = c + x^0.2 - x^(- x / 12) + 1/x - log(x)
test(3)
for i in 1:10
	println(test(ADVar(i, [1])).x, " ", test(ADVar(i, [1])).dx)
end

x=10
[[ test(x)::Float64 for x in 1:10] [ test(ADVar(a, [1])).x for a in 1:10] ]  
#  ok égalité

[ test(ADVar(x, [1])).dx::Vector{Float64} for x in 1:10] 
[ (test(ADVar(x+1e-4, [1])).x - test(ADVar(x, [1])).x)/1e-4 for x in 1:10]
#  ok égalité

randn((2,2)) * ADVar(1, [1])  # marche 
randn((2,2)) * [ADVar(1, [1]), ADVar(2, [0])]  # marche pas
randn((2,2)) * [ADVar(1, [1]), ADVar(2, [0])]  # marche pas

[ADVar(1, [1]), ADVar(2, [0])] * [ADVar(1, [1]), ADVar(2, [0])] 


@time begin
	for i in 1:100000
		test(5)
	end
end # 0.04 - 0.06 sec

@time begin
	a = ADVar(5., [1.])
	for i in 1:100000
		test(a)
	end
end # 0.40 sec, x 10

@time begin
	a = ADVar(5., [1., 0, 1, 2, 3, 5, -5])
	for i in 1:100000
		test(a)
	end
end # 0.48 sec, x 10

#
@time begin
	a = randn((2,2))
	for i in 1:100000
		a * 1 
	end
end # 0.01 - 0.025 sec

@time begin
	a = randn((2,2))
	for i in 1:100000
		a * ADVar(1, [1]) 
	end
end # 0.09, x 5

#
@time begin
	a = randn((2,2))
	b = randn((2,2))
	for i in 1:1000000
		a * 1 
	end
end # 0.12 sec

@time begin
	a = randn((2,2))
	b = [ADVar(1, [1]) ADVar(2, [0]) ; ADVar(3, [1]) ADVar(4, [0])]
	for i in 1:1000000
		a * ADVar(1, [1]) 
	end
end # 0.90, x8

@time begin
	a = [ADVar(1, [1]) ADVar(2, [0]) ; ADVar(3, [1]) ADVar(4, [0])]
	b = [ADVar(1, [1]) ADVar(2, [0]) ; ADVar(3, [1]) ADVar(4, [0])]
	for i in 1:1000000
		a * ADVar(1, [1]) 
	end
end # 1.90, x16


############### test lm ####################
require("Distributions.jl") # will also make this go away
using Distributions      # this will trigger require
import Distributions.Normal, Distributions.Gamma, Distributions.logpdf

	begin
		srand(1)
		n = 10000
		nbeta = 40
		X = [fill(1, (n,)) randn((n, nbeta-1))]
		beta0 = randn((nbeta,))
		Y = X * beta0 + randn((n,))
	end

	beta = ones(5)
	function loglik(beta)
		dimb::Int32

		dimb = numel(beta)

		tmp =   [ ADVar(beta[i], dimb, i) for i in 2:dimb]
		resid = Y - X * tmp
		ll = 0.0
		ll += logpdf(Gamma(2, 1), beta[1])
		# sigma ~ Gamma(2, 1)

		tmp = [ ADVar(beta[i], dimb, i) for i in 2:dimb]
		ll += sum( - tmp' * tmp)

		tmp = resid' * resid 
		ll += sum( - resid) / beta[1]
		# resid ~ Normal(0.0, sigma)

		ll
	end

	function loglik0(beta)
		dimb::Int32

		dimb = numel(beta)
		tmp = beta[2:dimb]
		resid = Y - X * tmp
		ll = 0.0
		ll += logpdf(Gamma(2, 1), beta[1])

		ll += sum( - tmp' * tmp)

		tmp = resid' * resid 
		ll += sum( - resid) / beta[1]

		ll
	end




delta = loglik(beta)

@time begin
	beta = ones(nbeta+1)
	for i in 1:100
		delta = loglik0(beta)
	end
end #  10000 * 40 : 0.003 sec

@time begin
	beta = ones(nbeta+1)
	for i in 1:10
		delta = loglik(beta)
	end
end #  10000 * 40 : 0.4 sec par appel, x100 par rapport à la version de base



beta += delta.dx *0.9

expexp(:([ ADVar(i, 5, 1),  ADVar(i, 5, 2),  ADVar(i, 5, 3),  ADVar(i, 5, 4)]))

loglik(ones(5))

sum([ ADVar(1, [3]) ADVar(1, [3])])	
		model = quote
		end
