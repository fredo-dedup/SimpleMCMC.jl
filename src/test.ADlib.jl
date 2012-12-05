# tests on ADlib.jl library

load("ADlib.jl")
using ADlib

############################################################################
#  New type definition + method defs for automated derivation
############################################################################

	@assert vec(ADVar(ones(Float64, (1, 1, 4))).v) == [1, 1, 1, 1]


	@assert size(ADVar(ones(Float64, (1, 1, 4))).v) == (1, 1, 4)
	@assert size(ADVar(ones(Float64, (1, 1, 4)))) == (1, 1, 4)
	@assert size(ADVar(ones(Float64, (1, 1, 4))), 1) == 1
	@assert size(ADVar(ones(Float64, (1, 1, 4))), 3) == 4


	@assert vec(ADVar(1.0, 5, 2).v) == [1., 0, 1, 0, 0, 0]
	@assert vec(ADVar(1, 5, 2).v) == [1., 0, 1, 0, 0, 0]


	@assert vec(ADVar(2., [1., 2, 3]).v) == [2., 1, 2, 3] 
	@assert vec(ADVar(2., [1, 2, 3]).v) == [2., 1, 2, 3] 
	@assert vec(ADVar(2, [1., 2, 3]).v) == [2., 1, 2, 3] 

	@assert reshape(ADVar([1., 2, 3]).v, 12) == [1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1] 
	@assert reshape(ADVar([1., 2, 3])[2:3,:], 8) == [1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1] 

	######### addition  ######################
	@assert vec((ADVar(4., [1., 0]) + ADVar(4., [0., 1])).v) == [8, 1, 1]
	@assert vec((ADVar(4., [1., 0]) + 4).v) == [8, 1, 0]
	@assert vec((3. + ADVar(3., [0., 2])).v) == [6, 0, 2]

	######### substraction ######################

	@assert  vec((- ADVar(4., [1., 0])).v) == [-4., -1, 0]
	@assert  vec((ADVar(4., [1., 0]) - ADVar(4., [1., 0])).v) == [0., 0, 0]
	@assert  vec((ADVar(4., [1., 0]) - 1.0).v) == [3., 1, 0]
	@assert  vec((3 - ADVar(4., [1., 0])).v) == [-1., -1, 0]

	######### multiplication  ######################

	a = ADVar(4., [1., 0])
	b = ADVar(2., [0., 1])

	@assert  vec((a * b).v) == [8., 2, 4]
	@assert  vec((a * 3).v) == [12., 3, 0]

	@assert  size((a * [1 2]).v) == (1, 2, 3)
	@assert  [(a * [1 2]).v[i] for i in 1:6] == [4, 8, 1, 2, 0, 0]
	@assert  size(([1, 2] * a).v) == (2, 1, 3)
	@assert  [([1, 2] * a).v[i] for i in 1:6] == [4, 8, 1, 2, 0, 0]

	######### division  ######################

	a = ADVar(4., [1., 0])
	b = ADVar(2., [0., 1])

	@assert  vec((a / b).v) == [2, 0.5, -1]
	@assert  vec((a / 4).v) == [1, 0.25, 0]
	@assert  vec((4. / a).v) == [1, -0.25, 0]

	######### power  ######################

	@assert vec((ADVar(3.0, [1., 0]) ^ 3.).v) == [27., 27, 0]
	@assert norm(vec((2.0 ^ ADVar(3., [1., 0])).v) -
	[8., 5.54518, 0]) < 1e-5
	@assert norm(vec((ADVar(3.0, [1., 0]) ^ ADVar(3.0, [0., 1])).v) -
		[27., 27, 29.6625] ) < 5e-5

	######### misc  ######################

	@assert norm( vec(log(ADVar(4., [1., 3])).v) - [1.386294, 0.25, 0.75]) < 1e-6

	########### complete function ##########

	c = 3
	test(x) = c + x^0.2 - x^(- x / 12) + 1. / x - log(x)
	
	reference = [ test(x)::Float64 for x in 1:10]
	calc = [ test(ADVar(a+0., [1.])).v[1] for a in 1:10]
	@assert norm( reference - calc) < 1e-9

	reference = [ test(ADVar(x+0., [1.])).v[2]::Float64 for x in 1:10]
	calc = [ ((test(x+1e-4) - test(x))/1e-4)::Float64 for x in 1:10]
	@assert norm( reference - calc) < 1e-3


########################### benchmarks scalar functions ###########################
function timefactor(ex::Expr, it::Integer, reft::Float64)
	tic()
	eval(quote
		for i in 1:($it)
			$ex
		end
	end)
	t = toq()
	t / reft
end
timefactor(ex::Expr, it::Integer) = timefactor(ex, it, 1.0)

timefactor(:(test(5.)), 1000000, 1.0) # 0.58

a = ADVar(5., [1.])
timefactor(:(test(a)), 10000, 0.58 / 100 * 2) # x60

ADVar(5., [1., 0, 1, 2, 3, 5, -5])
timefactor(:(test(a)), 10000, 0.58 / 100 * 8) # x14 (less penalty if many derivatives)

########################### benchmarks vector multiplication ###########################
a, b = randn((1, 100)), randn((100, 1))
timefactor(:(a*b), 100000) # 0.11 sec

a, b = randn((1, 100)), ADVar(ones(100, 1, 2))
timefactor(:(a*b), 10000, 0.11 / 10 * 2) # x4

a, b = ADVar(ones(100, 1, 2))', ADVar(ones(100, 1, 2))  #'
timefactor(:(a*b), 10000, 0.11 / 10 * 2) # x4.3

a, b = ADVar(ones(100, 1, 20))', ADVar(ones(100, 1, 20)) #'
timefactor(:(a*b), 10000, 0.11 / 10 * 21) # x4.3

########################### benchmarks matrix multiplication ###########################
a, b = randn((100,100)), randn((100,100))
timefactor(:(a*b), 10000) # 1.2 sec

a, b = randn((100,100)), ADVar(ones(100, 100, 2))
timefactor(:(a*b), 10000, 1.2 * 2) # x2.1

a, b = ADVar(ones(100, 100, 2)), ADVar(ones(100, 100, 2))
timefactor(:(a*b), 10000, 1.2 * 2) # x2.9

a, b = randn((100,100)), ADVar(ones(100, 100, 20))
timefactor(:(a*b), 1000, 1.2 / 10 * 21) # x2.1

a, b = ADVar(ones(100, 100, 20)), ADVar(ones(100, 100, 20))
timefactor(:(a*b), 1000, 1.2 / 10 * 21) # x3.6

########################### benchmarks matrix addition ###########################
a, b = randn((100,100)), randn((100,100))
timefactor(:(a+b), 10000) # .82 sec

a, b = randn((100,100)), ADVar(ones(100, 100, 2))
timefactor(:(a+b), 10000, 0.82 * 2) # x0.2 (weird !)

a, b = ADVar(ones(100, 100, 2)), ADVar(ones(100, 100, 2))
timefactor(:(a+b), 10000, 0.82 * 2) # x0.88 (weird !)

a, b = randn((100,100)), ADVar(ones(100, 100, 20))
timefactor(:(a+b), 10000, 0.82 * 21) # x0.02 (weird !)

a, b = ADVar(ones(100, 100, 20)), ADVar(ones(100, 100, 20))
timefactor(:(a+b), 10000, 0.82 * 21) # x0.84

########################### benchmarks matrix substraction ###########################
a, b = randn((100,100)), randn((100,100))
timefactor(:(a-b), 10000) # .91 sec

a, b = randn((100,100)), ADVar(ones(100, 100, 2))
timefactor(:(a-b), 10000, 0.91 * 2) # x0.82 

a, b = ADVar(ones(100, 100, 2)), ADVar(ones(100, 100, 2))
timefactor(:(a-b), 10000, 0.91 * 2) # x0.82

a, b = randn((100,100)), ADVar(ones(100, 100, 20))
timefactor(:(a-b), 10000, 0.91 * 21) # x0.61

a, b = ADVar(ones(100, 100, 20)), ADVar(ones(100, 100, 20))
timefactor(:(a-b), 10000, 0.91 * 21) # x0.80

############### benchmark linear regression ####################
require("Distributions.jl") # will also make this go away
using Distributions

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

		ll
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
		ll -= sum(resid' * resid) / beta[1]

		ll
	end

beta = ones(nbeta+1)

@assert abs(loglik0(beta) - loglik(beta).v[1]) < 1e-9

reference = [ loglik(beta).v[i]::Float64 for i in 2:41]
calc = [ (beta1 = copy(beta); beta1[i] += 1e-7; (loglik0(beta1) - loglik0(beta))/1e-7)::Float64 
				for i in 2:41]
@assert norm( reference - calc) < 1  # high tolerance as numerical derivation seems imprecise

ref_time = timefactor(:(loglik0(beta)), 10000) # 1.85 sec
timefactor(:(loglik(beta)), 100, ref_time / 100 * (nbeta+1)) # x10

