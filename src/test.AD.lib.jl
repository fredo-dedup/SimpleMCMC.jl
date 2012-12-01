load("AD.lib.jl")
import ADlib.*

load("AD.lib.2.jl")
import ADlib.ADVar, ADlib.isscalar
+
import Base.convert

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

randn((2,2)) * ADVar(1, [1])  # marche 
randn((2,2)) * [ADVar(1, [1]), ADVar(2, [0])]  # marche
randn((2,2)) * [ADVar(1, [1]), ADVar(2, [0])]  # marche

[ADVar(1, [1]), ADVar(2, [0])] * [ADVar(1, [1]), ADVar(2, [0])] 


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

#
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
end # 5.05 sec, x4.4 pas mal

@time begin
	a = randn((100,100))
	b = ADVar(ones(100, 100, 20)) 
	for i in 1:1000
		a * b
	end
end # 53.3 sec, x46

@time begin
	a = ADVar(ones(100, 100, 20)) 
	b = ADVar(ones(100, 100, 20)) 
	for i in 1:100
		a * b
	end
end # 91 sec, x79



#######################################################################
#   ADArray type definition
#######################################################################
b = [ADVar(randn(), [randn(), randn()]) for i in 1:2]

type ADArray
	dx::Array{Float64, 3}
end
convert(::Type{Vector{Float64}}, x::ADVar) = [x.x, x.dx]::Vector{Float64}
function ADArray(x::Array{ADVar})
	local n,m,d

	d = size(x[1].dx)[1] + 1 # nb of derivates + value
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

#######################################################################
#   multiplication definition
#######################################################################
function *(X::ADArray, Y::ADArray)
	d1, n1, m1 = size(X.dx)
	d2, n2, m2 = size(Y.dx)

	if m1 != n2
		error("Dimensions do not match")
	elseif d1 != d2
		error("Nb of derivatives do not match")
	end

	tmp = Array(Float64, (d1, n1, m2))
	Xv = reshape(sub(X.dx, 1, 1:n1, 1:m1), (n1, m1))
	Yv = reshape(sub(Y.dx, 1, 1:n2, 1:m2), (n2, m2))

	assign(tmp, reshape(Xv*Yv, (1, n1, m2)), 1, 1:n1, 1:m2)
	for i in 2:d1
		tmp2 = Xv * reshape(sub(Y.dx, i, 1:n2, 1:m2), (n2, m2)) +
			reshape(sub(X.dx, i, 1:n1, 1:m1), (n1, m1)) * Yv
		assign(tmp, reshape(tmp2, (1, n1, m2)), i, 1:n1, 1:m2)
	end
	ADArray(tmp)
end

function matmul2(X::ADArray, Y::ADArray)
	d1, n1, m1 = size(X.dx)
	d2, n2, m2 = size(Y.dx)

	if m1 != n2
		error("Dimensions do not match")
	elseif d1 != d2
		error("Nb of derivatives do not match")
	end

	tmp = Array(Float64, (n1, m2, d1))
	X2 = permute(X.dx, [2, 3, 1])
	Y2 = permute(Y.dx, [2, 3, 1])

	Xv = reshape(X2[1:(n1*m1)], (n1, m1))
	Yv = reshape(Y2[1:(n2*m2)], (n2, m2))

	tmp[1:(n1*m2)] = Xv * Yv
	i1, i2, i3 = 1, 1, 1
	for i in 2:d1
		i1 += n1*m1
		i2 += n2*m2
		i3 += n1*m2
		# println(i3:(i3+n1*m2-1)) 
		tmp[i3:(i3+n1*m2-1)] = Xv * reshape(Y2[i2:(i2+n2*m2-1)], (n2, m2)) +
			reshape(X2[i1:(i1+n1*m1-1)], (n1, m1)) * Yv 
	end
	ADArray(permute(tmp, [3, 1, 2]))
end

function matmul3(X::ADArray, Y::ADArray)
	d1, n1, m1 = size(X.dx)
	d2, n2, m2 = size(Y.dx)

	if m1 != n2
		error("Dimensions do not match")
	elseif d1 != d2
		error("Nb of derivatives do not match")
	end

	tmp = Array(Float64, (n1, m2, d1))
	# X2 = permute(X.dx, [2, 3, 1])
	# Y2 = permute(Y.dx, [2, 3, 1])
	X2, Y2 = X.dx, Y.dx
	Xv = reshape(X2[1:(n1*m1)], (n1, m1))
	Yv = reshape(Y2[1:(n2*m2)], (n2, m2))

	tmp[1:(n1*m2)] = Xv * Yv

	# tmp = Xv * reshape(Y2[(n2*m2+1):end], (n2, m2*(d1-1)))
	# tmp += reshape(Y2[(n2*m2+1):end], (n2, m2*(d1-1))) * Yv
	i1, i2, i3 = 1, 1, 1
	for i in 2:d1
		i1 += n1*m1
		i2 += n2*m2
		i3 += n1*m2
		# println(i3:(i3+n1*m2-1)) 
		tmp[i3:(i3+n1*m2-1)] = Xv * reshape(Y2[i2:(i2+n2*m2-1)], (n2, m2)) +
			reshape(X2[i1:(i1+n1*m1-1)], (n1, m1)) * Yv 
	end
	# ADArray(permute(tmp, [3, 1, 2]))
	ADArray(tmp)
end



#######################################################################
#  Benchmarks
#######################################################################
@time begin
	a = randn(10,10)
	b = randn(10,30)
	for i in 1:1000000
		a * b
	end
end # 0.037 sec

@time begin # mult naive
	a = [ADVar(randn(), [randn(), randn()]) for i in 1:10, j in 1:10]
	b = [ADVar(randn(), [randn(), randn()]) for i in 1:10, j in 1:10]
	for i in 1:10000
		a * b
	end
end # 6.12 sec, x165

@time begin # utlisation de array
	a = ADArray([ADVar(randn(), [randn(), randn()]) for i in 1:10, j in 1:10])
	b = ADArray([ADVar(randn(), [randn(), randn()]) for i in 1:10, j in 1:10])
	for i in 1:10000
		a * b
	end
end # 3.09, x81

@time begin # optim 1
	a = ADArray([ADVar(randn(), [randn(), randn()]) for i in 1:10, j in 1:10])
	b = ADArray([ADVar(randn(), [randn(), randn()]) for i in 1:10, j in 1:10])
	for i in 1:10000
		matmul2(a, b)
	end
end # 0.52, x14  c'est mieux !


@time begin # optim 1 sans permute
	a = ADArray([ADVar(randn(), [randn(), randn()]) for i in 1:10, j in 1:10])
	b = ADArray([ADVar(randn(), [randn(), randn()]) for i in 1:10, j in 1:10])
	for i in 1:100000
		matmul3(a, b)
	end
end # 0.23, x6.2

z = [randn() for i in 1:2, j in 1:3, k in 1:2]
ADArray(z)

b = [ADVar(randn(), [randn(), randn()]) for i in 1:2]
ADArray(b)

X = ADArray([ADVar(randn(), [randn(), randn()]) for i in 1:2])
Y = ADArray([ADVar(randn(), [randn(), randn()]) for i in 1:1, j in 1:2])

X*Y
matmul2(X, Y)




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

		tmp = Array(Float64, (dimb+1, 1, dimb))
		 [ ADVar(beta[i], dimb, i) for i in 2:dimb]
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
	for i in 1:1000
		delta = loglik0(beta)
	end
end #  10000 * 40 : 0.00032 sec par itération


@time begin
	beta = ones(nbeta+1)
	for i in 1:10
		delta = loglik(beta)
	end
#  10000 * 40 : 0.4 sec par appel, x100 par rapport à la version de base
end


beta += delta.dx *0.9

expexp(:([ ADVar(i, 5, 1),  ADVar(i, 5, 2),  ADVar(i, 5, 3),  ADVar(i, 5, 4)]))

loglik(ones(5))

sum([ ADVar(1, [3]) ADVar(1, [3])])	
		model = quote
		end
