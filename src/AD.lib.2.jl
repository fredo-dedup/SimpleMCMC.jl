module ADlib

	# using Base

	import Base.+, Base.*, Base./
	import Base.log, Base.(-), Base.^
	import Base.conj

	export +, *, /, log, -, ^, conj, ADVar

	type ADVar
		x::Float64
		dx::Vector{Float64}
	end

	ADVar(x::Float64, dim::Integer, index::Integer) = ADVar(x, [i==index ? 1.0 : 0.0 for i in 1:dim])
	ADVar(x::Int32, dim::Integer, index::Integer) = ADVar(convert(Float64,x), [i==index ? 1.0 : 0.0 for i in 1:dim])
	ADVar(x::Int64, dim::Integer, index::Integer) = ADVar(convert(Float64,x), [i==index ? 1.0 : 0.0 for i in 1:dim])
	ADVar(3.0, 4, 3)

	ADVar(x::Number, dx::Vector{Number}) =  ADVar(convert(Float64,x), convert(Vector{Float64}, dx))
	ADVar(x::Int32, dx::Vector{Int32}) =  ADVar(convert(Float64,x), convert(Vector{Float64}, dx))
	ADVar(x::Int64, dx::Vector{Int64}) =  ADVar(convert(Float64,x), convert(Vector{Float64}, dx))
	ADVar(x::Float64, dx::Vector{Int32}) =  ADVar(x, convert(Vector{Float64}, dx))
	ADVar(x::Float64, dx::Vector{Int64}) =  ADVar(x, convert(Vector{Float64}, dx))
	ADVar(x::Int32, dx::Vector{Float64}) =  ADVar(convert(Float64,x), dx)
	ADVar(x::Int64, dx::Vector{Float64}) =  ADVar(convert(Float64,x), dx)

	# @assert isequal(ADVar(3.0, [1., 2]), ADVar(3.0, [1., 2]))
	ADVar(3.0, [1, 2])
	ADVar(3, [1, 2])
	ADVar(3, [1., 2])

	# convert(::Type{ADVar}, x::Number) = ADVar(x, zeros(2))
	# promote_rule(::Type{Float64}, ::Type{ADVar}) = ADVar
		# promote_rule(::Type{Number}, ::Type{ADVar}) = ADVar

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

		@assert  (- ADVar(4., [1., 0])).x == -4.
		@assert  (- ADVar(4., [1., 0])).dx == [-1., 0]
		@assert  (ADVar(4., [1., 0]) - ADVar(4., [1., 0])).dx == [0., 0]
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

		function adadd(X::AbstractArray, Y::AbstractArray)
			@assert size(X) == size(Y)
			Z = similar(X, ADVar)
			for i = 1:numel(X)
		        Z[i] = X[i] + Y[i]
		    end
		    return Z
		end
		+(X::Matrix{ADVar}, Y::Matrix{ADVar}) = adadd(X, Y)
		+(X::Matrix, Y::Matrix{ADVar}) = adadd(X, Y)
		+(X::Matrix{ADVar}, Y::Matrix) = adadd(X, Y)

		+(X::Vector{ADVar}, Y::Matrix{ADVar}) = adadd(reshape(X, (size(X)[1], 1)), Y)
		+(X::Matrix{ADVar}, Y::Vector{ADVar}) = adadd(X, reshape(Y, (size(Y)[1], 1)))
		+(X::Vector{ADVar}, Y::Vector{ADVar}) = adadd(reshape(X, (size(X)[1], 1)), reshape(Y, (size(Y)[1], 1)))

		+(X::Vector, Y::Matrix{ADVar}) = adadd(reshape(X, (size(X)[1], 1)), Y)
		+(X::Matrix, Y::Vector{ADVar}) = adadd(X, reshape(Y, (size(Y)[1], 1)))
		+(X::Vector, Y::Vector{ADVar}) = adadd(reshape(X, (size(X)[1], 1)), reshape(Y, (size(Y)[1], 1)))

		+(X::Vector{ADVar}, Y::Matrix) = adadd(reshape(X, (size(X)[1], 1)), Y)
		+(X::Matrix{ADVar}, Y::Vector) = adadd(X, reshape(Y, (size(Y)[1], 1)))
		+(X::Vector{ADVar}, Y::Vector) = adadd(reshape(X, (size(X)[1], 1)), reshape(Y, (size(Y)[1], 1)))

		-(X::Matrix{ADVar}, Y::Matrix{ADVar}) = adadd(X, -1 * Y)
		-(X::Matrix, Y::Matrix{ADVar}) = adadd(X, -1 * Y)
		-(X::Matrix{ADVar}, Y::Matrix) = adadd(X, -1 * Y)

		-(X::Vector{ADVar}, Y::Matrix{ADVar}) = adadd(reshape(X, (size(X)[1], 1)), -1 * Y)
		-(X::Matrix{ADVar}, Y::Vector{ADVar}) = adadd(X, -1 * reshape(Y, (size(Y)[1], 1)))
		-(X::Vector{ADVar}, Y::Vector{ADVar}) = adadd(reshape(X, (size(X)[1], 1)), -1 * reshape(Y, (size(Y)[1], 1)))

		-(X::Vector, Y::Matrix{ADVar}) = adadd(reshape(X, (size(X)[1], 1)), -1 * Y)
		-(X::Matrix, Y::Vector{ADVar}) = adadd(X, -1 * reshape(Y, (size(Y)[1], 1)))
		-(X::Vector, Y::Vector{ADVar}) = adadd(reshape(X, (size(X)[1], 1)), -1 * reshape(Y, (size(Y)[1], 1)))

		-(X::Vector{ADVar}, Y::Matrix) = adadd(reshape(X, (size(X)[1], 1)), -1 * Y)
		-(X::Matrix{ADVar}, Y::Vector) = adadd(X, -1 * reshape(Y, (size(Y)[1], 1)))
		-(X::Vector{ADVar}, Y::Vector) = adadd(reshape(X, (size(X)[1], 1)), -1 * reshape(Y, (size(Y)[1], 1)))


	########### array products 1 ####################
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

	########### array products 2 ####################
	X = randn(4) * ADVar(2, [1, 1, 0])
	Y = randn((1,4))

		function admul(X::Matrix, Y::Matrix)
			# X = ndims(X) == 1 ? reshape(X, (size(X)[1], 1)) : X
			# Y = ndims(Y) == 1 ? reshape(Y, (size(Y)[1], 1)) : Y

			@assert ndims(X)==2 & ndims(Y)==2
		    mx, nx = size(X)
		    my, ny = size(Y)

		    @assert nx == my
		    Z = Array(ADVar, (mx, ny))
		    for i = 1:mx
			    for j = 1:ny
			    	# Z[i, j] = X[i,:] * Y[:, j]
			    	Z[i, j] = sum([ X[i, k] * Y[k, j] for k in 1:nx])
			    end
			end
		    return Z
		end
		*(X::Matrix{ADVar}, Y::Matrix{ADVar}) = admul(X, Y)
		*(X::Matrix, Y::Matrix{ADVar}) = admul(X, Y)
		*(X::Matrix{ADVar}, Y::Matrix) = admul(X, Y)

		*(X::Vector{ADVar}, Y::Matrix{ADVar}) = admul(reshape(X, (size(X)[1], 1)), Y)
		*(X::Matrix{ADVar}, Y::Vector{ADVar}) = admul(X, reshape(Y, (size(Y)[1], 1)))
		*(X::Vector{ADVar}, Y::Vector{ADVar}) = admul(reshape(X, (size(X)[1], 1)), reshape(Y, (size(Y)[1], 1)))

		*(X::Vector, Y::Matrix{ADVar}) = admul(reshape(X, (size(X)[1], 1)), Y)
		*(X::Matrix, Y::Vector{ADVar}) = admul(X, reshape(Y, (size(Y)[1], 1)))
		*(X::Vector, Y::Vector{ADVar}) = admul(reshape(X, (size(X)[1], 1)), reshape(Y, (size(Y)[1], 1)))

		*(X::Vector{ADVar}, Y::Matrix) = admul(reshape(X, (size(X)[1], 1)), Y)
		*(X::Matrix{ADVar}, Y::Vector) = admul(X, reshape(Y, (size(Y)[1], 1)))
		*(X::Vector{ADVar}, Y::Vector) = admul(reshape(X, (size(X)[1], 1)), reshape(Y, (size(Y)[1], 1)))

		[1 2] * [ADVar(1, [1]), ADVar(2, [0])] 
		[1 2 ; 3 4] * [ADVar(1, [1]), ADVar(2, [0])] 
		[1 2 ; 3 4] * [ADVar(1, [1]) ADVar(2, [0]) ; ADVar(3, [1]) ADVar(4, [0])] 
		[ADVar(1, [1]), ADVar(2, [0])] * [1 2] 
		[ADVar(1, [1]) ADVar(2, [0])] * [1 2 ; 3 4]
		[ADVar(1, [1]) ADVar(2, [0]) ; ADVar(3, [1]) ADVar(4, [0])] * [1 2 ; 3 4]

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
		^(x::ADVar, y::Integer) = x ^ ADVar(y, zeros(size(x.dx))) 
		^(x::ADVar, y::Any) = x ^ ADVar(y, zeros(size(x.dx))) 
		^(x::Any, y::ADVar) = ADVar(x, zeros(size(y.dx))) ^ y

		@assert (ADVar(3.0, [1., 0]) ^ 3).x == 27.
		@assert (ADVar(3.0, [1., 0]) ^ 3).dx == [27., 0]
		@assert (2.0 ^ ADVar(3.0, [1., 0])).x == 8.
		@assert norm((2.0 ^ ADVar(3.0, [1., 0])).dx - [5.54518, 0] ) < 1e-5

	######### misc  ######################
		log(x::ADVar) = ADVar(log(x.x), x.dx / x.x)
		conj(x::ADVar) = x

		@assert abs( (log(ADVar(4, [1, 3]))).x - 1.386294) < 1e-6
		@assert norm( (log(ADVar(4, [1, 3]))).dx - [0.25, 0.75]) < 1e-6

end

