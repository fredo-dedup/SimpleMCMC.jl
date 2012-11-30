load("AD.lib.jl")
import ADlib.*

load("AD.lib.2.jl")
import ADlib.ADVar


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
	for i in 1:1000000
		test(5)
	end
end # 0.75 sec

@time begin
	a = ADVar(5., [1.])
	for i in 1:1000000
		test(a)
	end
end # 6.7 sec, x9

@time begin
	a = ADVar(5., [1., 0, 1, 2, 3, 5, -5])
	for i in 1:1000000
		test(a)
	end
end # 9.2 sec, x12

#
@time begin
	a = randn((2,2))
	for i in 1:1000000
		a * 1 
	end
end # 0.21 sec

@time begin
	a = randn((2,2))
	for i in 1:1000000
		a * ADVar(1, [1]) 
	end
end # 1.714, x8

#
@time begin
	a = randn((2,2))
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

# matrix multiply 2
@time begin
	a = randn((100,100))
	b = randn(100)
	for i in 1:100000
		a * b
	end
end # 1.6 sec

@time begin
	a = randn((100,100))
	b = [ADVar(randn(), [randn(), randn()]) for i in 1:100]
	for i in 1:10000
		a * b
	end
end # 626 sec, x391 !!!

type ADVar2 <: Vector{Float64}

end

b = [ADVar(randn(), [randn(), randn()]) for i in 1:2]

type ADArray
	dx::Array{Float64, 3}
end

convert(::Type{Vector{Float64}}, x::ADVar) = [x.x, x.dx]::Vector{Float64}
# convert(Vector{Float64}, b[1])
# b

x = b
function ADArray(x::Array{ADVar})
	local n,m,d

	d = size(x[1].dx)[1] + 1 # nb of derivates
	if ndims(x) == 1
		n, m = size(x)[1], 1
	elseif ndims(x) == 2
		n, m = size(x)
		x = reshape(x, (n*m, 1))
	else
		error("up to 2 dims supported")
	end
	tmp = Array(Float64, (d*n*m))

	x = [convert(Vector{Float64}, x[i]) for i in 1:(n*m)]
	for i in 1:numel(x)
		tmp[i*3-2:i*3] = convert(Vector{Float64}, x[i])
	end
	
	ADArray(reshape(tmp, (d, n, m)))
end


z = [randn() for i in 1:2, j in 1:3, k in 1:2]
ADArray(z)

b = [ADVar(randn(), [randn(), randn()]) for i in 1:2]
ADArray(b)

X = ADArray([ADVar(randn(), [randn(), randn()]) for i in 1:2])
Y = ADArray([ADVar(randn(), [randn(), randn()]) for i in 1:1, j in 1:2])

function *(X::ADArray, Y::ADArray)
	d1, n1, m1 = size(X.dx)
	d2, n2, m2 = size(Y.dx)

	if m1 != n2
		error("Dimensions do not match")
	elseif d1 != d2
		error("Nb of derivatives do not match")
	end

	tmp = Array(Float64, (d1, n1, m2))
	tmp[1,:, :] = slicedim(X.dx, 1, 1) [1, :, :]) * squeeze(Y.dx[1, :, :])
end

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
