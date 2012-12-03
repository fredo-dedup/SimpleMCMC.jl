# load("AD.lib.jl")
# import ADlib.*
import Base


load("AD.lib.2.jl")
# using ADlib
import ADlib.ADVar, ADlib.isscalar, ADlib.size
import ADlib.+, ADlib.*, ADlib.-, ADlib.^
+
^
import Base.convert

ADVar(1., [2., 3]) + 5.0
ADVar(1, [2, 3]) + 5.0

########## scrapbook  #######################
c = 3
test(x) = c + x^0.2 - x^(- x / 12) + 1. / x - log(x)
test(3)

x=10
[[ test(x)::Float64 for x in 1:10] [ test(ADVar(a+0., [1.])).v[1] for a in 1:10] ]  
#  ok égalité

hcat([ [test(ADVar(x+0., [1.])).v[2]]::Vector{Float64} for x in 1:10],
	[ [(test(x+1e-4) - test(x))/1e-4]::Vector{Float64} for x in 1:10])
#  ok égalité

########################### benchmarks scalar functions ###########################
macro timefactor(ex::Expr, it::Integer)
	tic()
	eval(quote
		for i in 1:($it)
			$ex
		end
	end)
	t = toc()
	t / ref_time
end
it= 1000000

ref_time = 1.0
@timefactor test(5.) 1000000  # 0.78

a = ADVar(5., [1.])
(@timefactor test(a) 10000) * 100 / 2 # x36

ADVar(5., [1., 0, 1, 2, 3, 5, -5])
(@timefactor test(a) 10000) * 100 / 8 # x9

########################### benchmarks vector multiplication ###########################
a, b, ref_time = randn((1, 100)), randn((100, 1)), 1.0
ref_time = @timefactor a*b 100000 # 0.14 sec

a, b = randn((1, 100)), ADVar(ones(100, 1, 2))
(@timefactor a*b 10000) * 10 / 2 # x5.0

a, b = ADVar(ones(100, 1, 2))', ADVar(ones(100, 1, 2))  #'
(@timefactor a*b 10000) * 10 / 2 # x50

a, b = ADVar(ones(100, 1, 20))', ADVar(ones(100, 1, 20)) #'
(@timefactor a*b 1000) * 100 / 21 # x8.4

########################### benchmarks matrix multiplication ###########################
a, b, ref_time = randn((100,100)), randn((100,100)), 1.0
ref_time = @timefactor a*b 10000 # 2.90 sec

a, b = randn((100,100)), ADVar(ones(100, 100, 2))
(@timefactor a*b 1000) * 10 / 2 # x2.2

a, b = ADVar(ones(100, 100, 2)), ADVar(ones(100, 100, 2))
(@timefactor a*b 1000) * 10 / 2 # x3.4

a, b = randn((100,100)), ADVar(ones(100, 100, 20))
(@timefactor a*b 100) * 100 / 21 # x2.1

a, b = ADVar(ones(100, 100, 20)), ADVar(ones(100, 100, 20))
(@timefactor a*b 100) * 100 / 21 # x4.1

########################### benchmarks matrix addition ###########################
a, b, ref_time = randn((100,100)), randn((100,100)), 1.0
ref_time = @timefactor a+b 10000 # 1.46 sec

a, b = randn((100,100)), ADVar(ones(100, 100, 2))
(@timefactor a+b 10000) / 2 # x0.12 !!

a, b = ADVar(ones(100, 100, 2)), ADVar(ones(100, 100, 2))
(@timefactor a+b 10000) / 2 # x0.94 

a, b = randn((100,100)), ADVar(ones(100, 100, 20))
(@timefactor a+b 10000) / 21 # x0.01

a, b = ADVar(ones(100, 100, 20)), ADVar(ones(100, 100, 20))
(@timefactor a+b 1000) * 10 / 21 # x0.97

########################### benchmarks matrix substraction ###########################
a, b, ref_time = randn((100,100)), randn((100,100)), 1.0
ref_time = @timefactor a-b 10000 # 1.48 sec

a, b = randn((100,100)), ADVar(ones(100, 100, 2))
(@timefactor a-b 10000) / 2 # x0.95 !!

a, b = ADVar(ones(100, 100, 2)), ADVar(ones(100, 100, 2))
(@timefactor a-b 10000) / 2 # x0.93 !!

a, b = randn((100,100)), ADVar(ones(100, 100, 20))
(@timefactor a-b 1000) * 10 / 21 # x0.89

a, b = ADVar(ones(100, 100, 20)), ADVar(ones(100, 100, 20))
(@timefactor a-b 1000) * 10 / 21 # x0.97


############### test lm ####################
require("Distributions.jl") # will also make this go away
# using Distributions      # this will trigger require
import Distributions.Normal, Distributions.Gamma, Distributions.logpdf

	begin
		srand(1)
		n = 10000
		nbeta = 40
		X = [fill(1, (n,)) randn((n, nbeta-1))]
		beta0 = randn((nbeta,))
		Y = X * beta0 + randn((n,))
	end

	function loglik(beta::Array{Float64,1})
		local dimb::Int32
		local tmpa::ADVar, resid::ADVar
		local ll::ADVar

		dimb = numel(beta)

		ll = ADVar(0.0, zeros(Float64, dimb-1))
		ll += logpdf(Gamma(2, 1), beta[1])

		tmpa = ADVar(beta[2:end])
		ll += sum( - tmpa' * tmpa)

		resid = Y - X * tmpa
		ll -= sum(resid' * resid) / beta[1]

		return ll
	end

	function loglik0(beta::Array{Float64,1})
		local dimb::Int32
		local tmp::Array{Float64,1}
		local resid::Array{Float64, 1}
		local ll::Float64

		dimb = numel(beta)
		ll = 0.0
		ll += logpdf(Gamma(2, 1), beta[1])

		tmp = beta[2:dimb]
		ll += sum( - tmp' * tmp)
		
		resid = Y - X * tmp
		tmp = resid' * resid 
		ll += sum( - tmp) / beta[1]

		ll
	end



beta = ones(nbeta+1)

loglik0(beta)
loglik(beta)

ref_time = 1.0
ref_time = @timefactor loglik0(beta) 1000 # 0.74 sec
(@timefactor loglik(beta) 10) * 100 / (nbeta+1) # x4.15 !!

dn = [ (beta1=copy(beta) ; beta1[i] += 1e-4; (loglik0(beta1) - loglik0(beta))*1e4::Float64 ) for i in 2:size(beta,1)]	
vals = loglik(beta)
[ [dn[i-1], vals.v[i]] for i in 2:size(beta,1)]	# ok c'est égal









ref_time = 1.0
a, b = randn((100,100)), randn((100,100))
@timefactor a*b 10000 # 3.25 sec

a = randn((100,100))
@timefactor a' 10000 # 9.45 sec

a = randn((10000,1))
@timefactor a' 10000 # 12.6 sec

a = randn((1,10000))
@timefactor a' 10000 # 12.7 sec

a = randn((100,100))
@timefactor reshape(a, (200, 50)) 1000000 # 0.0034 sec, 2500 fois plus rapide !

a = randn((1,10000))
@timefactor reshape(a, (200, 50)) 1000000 # 0.0033 sec, 4000 fois plus rapide !

a = randn((10000,1))
@timefactor reshape(a, (200, 50)) 1000000 # 0.0033 sec, 4000 fois plus rapide !

