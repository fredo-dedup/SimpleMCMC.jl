# load("AD.lib.jl")
# import ADlib.*
import Base


load("AD.lib.2.jl")
using ADlib
import ADlib.ADVar, ADlib.isscalar, ADlib.size
+
import Base.convert

ADVar(1., [2., 3]) + 5.0
ADVar(1, [2, 3]) + 5.0

########## scrapbook  #######################
c = 3
test(x) = c + x^0.2 - x^(- x / 12) + 1. / x - log(x)
test(3)
for i in 1:10
	println(test(ADVar(i, [1])).x, " ", test(ADVar(i, [1])).dx)
end

x=10
[[ test(x) for x in 1:10] [ test(ADVar(a+0., [1.])).v[1] for a in 1:10] ]  
#  ok égalité

hcat([ test(ADVar(x+0., [1.])).v[2]::Vector{Float64} for x in 1:10],
	[ [(test(x+1e-4) - test(x))/1e-4] for x in 1:10])
#  ok égalité

########################### benchmarks scalar functions ###########################
@time begin
	for i in 1:1000000
		test(5)
	end
end # 0.53 sec

@time begin
	a = ADVar(5., [1.])
	for i in 1:10000
		test(a)
	end
end # 63 sec, x118

@time begin
	a = ADVar(5., [1., 0, 1, 2, 3, 5, -5])
	for i in 1:10000
		test(a)
	end
end # 88 sec, x166

########################### benchmarks matrix multiplication ###########################
@time begin
	a = randn((100,100))
	b = randn((100,100))
	for i in 1:10000
		a * b
	end
end # 1.15 sec

@time begin
	a = randn((100,100))
	b = ADVar(ones(100, 100, 2)) 
	for i in 1:10000
		a * b
	end
end # 5.05 sec, x4.4 pas mal, x

@time begin
	a = randn((100,100))
	b = ADVar(ones(100, 100, 20)) 
	for i in 1:1000
		a * b
	end
end # 53.3 sec, x46, x2.3 compared to numeric derivation

@time begin
	a = ADVar(ones(100, 100, 20)) 
	b = ADVar(ones(100, 100, 20)) 
	for i in 1:100
		a * b
	end
end # 91 sec, x79, x4 compared to numeric derivation

########################### benchmarks matrix addition ###########################
@time begin
	a = randn((100,100))
	b = randn((100,100))
	for i in 1:10000
		a + b
	end
end # 0.78 sec

@time begin
	a = randn((100,100))
	b = ADVar(ones(100, 100, 2)) 
	for i in 1:100
		a + b
	end
end 
# 5.05 sec, x4.4 pas mal, x

@time begin
	a = randn((100,100))
	b = ADVar(ones(100, 100, 20)) 
	for i in 1:1000
		a * b
	end
end # 53.3 sec, x46, x2.3 compared to numeric derivation

@time begin
	a = ADVar(ones(100, 100, 20)) 
	b = ADVar(ones(100, 100, 20)) 
	for i in 1:100
		a * b
	end
end # 91 sec, x79, x4 compared to numeric derivation





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

	function loglik(beta)
		dimb::Int32

		dimb = numel(beta)

		tmp = Array(Float64, (dimb-1, 1, dimb+1))
		tmp[:,1,1] = reshape(beta[2:end], (dimb-1, 1, 1))
		for i in 1:(dimb-1)
			tmp[i,1,i+2] = 1.
		end
		tmp = ADVar(tmp)

		resid = Y - X * tmp
		ll = 0.0
		ll += logpdf(Gamma(2, 1), beta[1])
		# sigma ~ Gamma(2, 1)

		# tmp = [ ADVar(beta[i], dimb, i) for i in 2:dimb]
		ll += sum( - tmp' * tmp)

		tmp2 = resid'
		tmp4 = tmp2 * resid 
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



beta = ones(nbeta+1)

loglik(beta)
loglik0(beta)

@time begin
	beta = ones(nbeta+1)
	for i in 1:1000
		delta = loglik0(beta)
	end
end 
#  10000 * 40 : 0.00029 sec par itération

@time begin
	beta = ones(nbeta+1)
	for i in 1:10
		delta = loglik(beta)
	end
end
#  10000 * 40 : 0.84 sec par iteration, x2897 argh...

