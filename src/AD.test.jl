
type ADVar
	x::Float64
	dx::Vector{Float64}
end
ADVar(x::Float64, dim::Integer, index::Integer) = ADVar(x, [i==index ? 1.0 : 0.0 for i in 1:dim])
ADVar(x::Int32, dim::Integer, index::Integer) = ADVar(convert(Float64,x), [i==index ? 1.0 : 0.0 for i in 1:dim])
ADVar(3.0, 4, 3)

ADVar(x::Number, dx::Vector{Number}) =  ADVar(convert(Float64,x), convert(Vector{Float64}, dx))
ADVar(x::Int32, dx::Vector{Int32}) =  ADVar(convert(Float64,x), convert(Vector{Float64}, dx))
ADVar(x::Float64, dx::Vector{Int32}) =  ADVar(x, convert(Vector{Float64}, dx))
ADVar(x::Int32, dx::Vector{Float64}) =  ADVar(convert(Float64,x), dx)

ADVar(3.0, [1., 2])
ADVar(3.0, [1, 2])
ADVar(3, [1, 2])
ADVar(3, [1., 2])

convert(::Type{ADVar}, x::Number) = ADVar(x, zeros(2))
promote_rule(::Type{Float64}, ::Type{ADVar}) = ADVar

###
	promote_rule(::Type{Number}, ::Type{ADVar}) = ADVar

	ADVar(4., [3.])

	convert(Number, 4.5)
	typeof(4) <: Number
	is(4, Integer)

	a = ADVar(3, 3, 1)  # error
	a = ADVar(3.0, 3, 1)
	a.x
	a.dx
	a
	x

######### addition  ######################
	+(x::ADVar, y::ADVar) = ADVar(x.x + y.x, x.dx + y.dx)
	+(x::ADVar, y::Any) = ADVar(x.x + y, x.dx)
	+(x::Any, y::ADVar) = +(y, x)

	@assert  (ADVar(4., [1., 0]) + ADVar(4., [1., 0])).x == 8.
	@assert  (ADVar(4., [1., 0]) + ADVar(4., [1., 0])).dx == [2., 0]
	@assert  (ADVar(4., [1., 0]) + 1.).x == 5.
	@assert  (ADVar(4., [1., 0]) + 4.).dx == [1., 0]
	@assert  (1. + ADVar(4., [1., 0])).x == 5.
	@assert  (4. + ADVar(4., [1., 0])).dx == [1., 0]

######### substraction ######################
	-(x::ADVar) = ADVar(-x.x, -x.dx)  # unary
	-(x::ADVar, y::ADVar) = ADVar(x.x - y.x, x.dx - y.dx)
	-(x::ADVar, y::Any) = ADVar(x.x - y, x.dx)
	-(x::Any, y::ADVar) = -(y - x)

	@assert  (ADVar(4., [1., 0]) - ADVar(4., [1., 0])).x == 0.
	@assert  (ADVar(4., [1., 0]) - ADVar(4., [1., 0])).dx == [0., 0]
	@assert  (ADVar(4., [1., 0]) - 1.).x == 3.
	@assert  (ADVar(4., [1., 0]) - 4.).dx == [1., 0]
	@assert  (1. - ADVar(4., [1., 0])).x == -3.
	@assert  (4. - ADVar(4., [1., 0])).dx == [-1., 0]

######### multiplication  ######################
	*(x::ADVar, y::ADVar) = ADVar(x.x * y.x, x.dx*y.x + y.dx*x.x)
	*(x::ADVar, y::Any) = ADVar(x.x*y, x.dx*y)  
	*(x::Any, y::ADVar) = *(y, x) # not correct for matrices !

	@assert  (ADVar(4., [1., 0]) * ADVar(4., [1., 0])).x == 16.
	@assert  (ADVar(4., [1., 0]) * ADVar(4., [0., 1])).dx == [4., 4]
	@assert  (ADVar(4., [1., 0]) * ADVar(4., [1., 0])).dx == [8., 0]
	@assert  (ADVar(4., [1., 0]) * 1.).x == 4.
	@assert  (ADVar(4., [1., 0]) * 4.).dx == [4., 0]
	@assert  (1. * ADVar(4., [1., 0])).x == 4.
	@assert  (4. * ADVar(4., [1., 0])).dx == [4., 0]

	#  x  + y^2 avec (x=4 et y=2)
	(ADVar(4., [1., 0]) + (ADVar(2., [0., 1]) * ADVar(2., [0., 1]))).x  # 8
	(ADVar(4., [1., 0]) + (ADVar(2., [0., 1]) * ADVar(2., [0., 1]))).dx #  [1.  4]

	#  x * y^2 avec (x=2 et y=3)
	(ADVar(2., [1., 0]) * (ADVar(3., [0., 1]) * ADVar(3., [0., 1]))).x  # 18
	(ADVar(2., [1., 0]) * (ADVar(3., [0., 1]) * ADVar(3., [0., 1]))).dx #  [9.  12.]

########### array sums ####################
	function +(X::AbstractArray{ADVar}, a::Number)
		for i = 1:numel(X)
	        X[i] = X[i] + a
	    end
	    return X
	end
	+(a::Number, X::AbstractArray{ADVar}) = X + a

	function +(X::AbstractArray, a::ADVar)
		Y = similar(X, ADVar)
		for i = 1:numel(X)
	        Y[i] = X[i] + a
	    end
	    return Y
	end
	+(a::ADVar, X::AbstractArray) = X + a

	[ADVar(1, [1]) ADVar(2, [0])] + 4.5
	4.5 + [ADVar(1, [1]), ADVar(2, [0])]

	[1., 2.] + ADVar(2, [0])
	[ADVar(1, [1]) ADVar(2, [0])] + ADVar(2, [0])
	ADVar(2, [0, 0.5]) + [ADVar(1, [1, 0]), ADVar(2, [0, 1])]

########### array products ####################
	function *(X::AbstractArray{ADVar}, a::Number)
		for i = 1:numel(X)
	        X[i] = X[i] * a
	    end
	    return X
	end
	*(a::Number, X::AbstractArray{ADVar}) = X * a

	function *(X::AbstractArray, a::ADVar)
		Y = similar(X, ADVar)
		for i = 1:numel(X)
	        Y[i] = X[i] * a
	    end
	    return Y
	end
	*(a::ADVar, X::AbstractArray) = X * a

	[ADVar(1, [1]) ADVar(2, [0])] * 4.5
	4.5 * [ADVar(1, [1]), ADVar(2, [0])]

	[1., 2.] * ADVar(2, [0])
	[ADVar(1, [1]) ADVar(2, [0])] * ADVar(2, [0])
	ADVar(2, [0, 0.5]) * [ADVar(1, [1, 0]), ADVar(2, [0, 1])]

######### division  ######################
	/(x::ADVar, y::ADVar) = ADVar(x.x / y.x, (x.dx*y.x - y.dx*x.x) / (y.x * y.x))
	/(x::ADVar, y::Any) = ADVar(x.x / y, x.dx / y) 
	/(x::Any, y::ADVar) = ADVar(x / y.x, - y.dx * (x / (y.x * y.x)))

	@assert  (ADVar(4., [1., 0]) / ADVar(2., [1., 0])).x == 2.
	@assert  (ADVar(4., [1., 0]) / ADVar(2., [0., 1])).dx == [.5, -1]
	@assert  (ADVar(4., [1., 0]) / 2.).x == 2.
	@assert  (ADVar(4., [1., 0]) / 2.).dx == [.5, 0]
	@assert  (1. / ADVar(4., [1., 0])).x == 0.25
	@assert  (4. / ADVar(4., [0., 1])).dx == [0., -.25]

######### power  ######################
	^(x::ADVar, y::ADVar) = ADVar(x.x ^ y.x, (y.x * x.x^(y.x-1.)) * x.dx + (log(x.x) * x.x^y.x) * y.dx)
	^(x::ADVar, y::Any) = x ^ ADVar(y, zeros(size(x.dx))) 
	^(x::Any, y::ADVar) = ADVar(x, zeros(size(y.dx))) ^ y

	@assert (ADVar(3.0, [1., 0]) ^ 3).x == 27.
	@assert (ADVar(3.0, [1., 0]) ^ 3).dx == [27., 0]
	@assert (2.0 ^ ADVar(3.0, [1., 0])).x == 8.
	@assert norm((2.0 ^ ADVar(3.0, [1., 0])).dx - [5.54518, 0] ) < 1e-5

######### misc  ######################
	log(x::ADVar) = ADVar(log(x.x), x.dx / x.x)

	log(ADVar(4, [1, 3]))

########## scrapbook  #######################
c = 46
test(x) = c + x^2 - x^(x / 12) + 1/x

[ test(x) for x in 1:10]
[ test(ADVar(x, [1])).x for x in 1:10] #  ok égalité
[ test(ADVar(x, [1])).dx for x in 1:10]
[ (test(ADVar(x+1e-4, [1])).x - test(ADVar(x, [1])).x)/1e-4 for x in 1:10]  # ok égalité


randn((2,2)) * ADVar(1, [1])  # marche 
randn((2,2)) * [ADVar(1, [1]), ADVar(2, [0])]  # marche pas
randn((2,2)) * [ADVar(1, [1]), ADVar(2, [0])]  # marche pas

[ADVar(1, [1]), ADVar(2, [0])] * [ADVar(1, [1]), ADVar(2, [0])] 


############### test lm ####################
	begin
		srand(1)
		n = 10
		nbeta = 4
		X = [fill(1, (n,)) randn((n, nbeta-1))]
		beta0 = randn((nbeta,))
		Y = X * beta0 + randn((n,))
	end

	function loglik(beta)
		dimb::Int32

		dimb = numel(beta)

		resid = Y - X * [ ADVar(beta[i], dimb, i) for i in 2:dimb]
		ll = 0.0
		ll += logpdf(Gamma(2, 1), beta[1])
		# sigma ~ Gamma(2, 1)

		tmp = [ ADVar(beta[i], 41, i) for i in 2:dimb]
		ll += sum( - tmp' * tmp)

		tmp = resid' * resid 
		ll += sum( - resid / sigma)
		# resid ~ Normal(0.0, sigma)

		ll
	end

expexp(:([ ADVar(i, 5, 1),  ADVar(i, 5, 2),  ADVar(i, 5, 3),  ADVar(i, 5, 4)]))

loglik(ones(5))

sum([ ADVar(1, [3]) ADVar(1, [3])])	
		model = quote
		end
