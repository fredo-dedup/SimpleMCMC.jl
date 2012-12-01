############################################################################
#  New type definition + method defs for automated derivation
############################################################################

module ADlib

	using Base

	import Base.+, Base.*, Base./
	import Base.log, Base.(-), Base.^
	import Base.conj, Base.ctranspose

	# supported operators
	export +, *, /, log, -, ^, conj, ADVar
	export isscalar

	type ADVar
		v::Array{Float64,3}
	end

	isscalar(x::ADVar) = size(x.v)[1:2] == (1, 1)

	function ADVar(x::Any, dim::Integer, index::Integer)
	 	tmp = Array(Float64, (1,1,dim+1))
	 	tmp[1:numel(tmp)] = [x, [i==index ? 1.0 : 0.0 for i in 1:dim]]
	 	ADVar(tmp)
	end

	function ADVar(x::Any, d::Vector{Any})
	 	tmp = Array(Float64, (1, 1, length(d)+1))
	 	tmp[1:numel(tmp)] = [x, d[1:end]]
	 	ADVar(tmp)
	end
	function ADVar(x::Float64, d::Vector{Float64})
	 	tmp = Array(Float64, (1, 1, length(d)+1))
	 	tmp[1:numel(tmp)] = [x, d[1:end]]
	 	ADVar(tmp)
	end

# import ADlib.ADVar, ADlib.isscalar
# import Base.+, Base 

	# @assert vec(ADVar(Array(Float64, (1, 1, 4))).v) == [0, 0, 0, 0]
	# @assert ADVar(fill(0, (2, 3, 2))).v == [0 for i in 1:2, j in 1:3, k in 1:2] # fail

	@assert vec(ADVar(1.0, 5, 2).v) == [1., 0, 1, 0, 0, 0]
	@assert vec(ADVar(1, 5, 2).v) == [1., 0, 1, 0, 0, 0]

	@assert vec(ADVar(2., [1., 2, 3]).v) == [2., 1, 2, 3] 
	# @assert vec(ADVar(2., [1., 2, 3]).v) == [2., 1, 2, 3] 


	ADVar(x::Int64, dim::Integer, index::Integer) = ADVar(convert(Float64,x), [i==index ? 1.0 : 0.0 for i in 1:dim])

	# ADVar(x::Number, dx::Vector{Number}) =  ADVar(convert(Float64,x), convert(Vector{Float64}, dx))
	# ADVar(x::Int32, dx::Vector{Int32}) =  ADVar(convert(Float64,x), convert(Vector{Float64}, dx))
	# ADVar(x::Int64, dx::Vector{Int64}) =  ADVar(convert(Float64,x), convert(Vector{Float64}, dx))
	# ADVar(x::Float64, dx::Vector{Int32}) =  ADVar(x, convert(Vector{Float64}, dx))
	# ADVar(x::Float64, dx::Vector{Int64}) =  ADVar(x, convert(Vector{Float64}, dx))
	# ADVar(x::Int32, dx::Vector{Float64}) =  ADVar(convert(Float64,x), dx)
	# ADVar(x::Int64, dx::Vector{Float64}) =  ADVar(convert(Float64,x), dx)

	######### addition  ######################
		+(x::ADVar, y::ADVar) = ADVar(x.v + y.v)
		function +(x::ADVar, y::Any)  #TODO rajouter matrices
			for i in 1:(size(x.v,1)*size(x.v,2))
				x.v[i] += y
			end
			x
		end
		+(x::Any, y::ADVar) = +(y, x)

		@assert vec((ADVar(4., [1., 0]) + ADVar(4., [0., 1])).v) == [8, 1, 1]
		@assert vec((ADVar(4., [1., 0]) + 4).v) == [8, 1, 0]
		@assert vec((3. + ADVar(3., [0., 2])).v) == [6, 0, 2]

	######### substraction ######################
		-(x::ADVar) = ADVar(-x.v)  # unary
		-(x::ADVar, y::ADVar) = ADVar(x.v - y.v)
		function -(x::ADVar, y::Any) #TODO rajouter matrices
			for i in 1:(size(x.v,1)*size(x.v,2))
				x.v[i] -= y
			end
			x
		end
		-(x::Any, y::ADVar) = -(y - x)

		@assert  vec((- ADVar(4., [1., 0])).v) == [-4., -1, 0]
		@assert  vec((ADVar(4., [1., 0]) - ADVar(4., [1., 0])).v) == [0., 0, 0]
		@assert  vec((ADVar(4., [1., 0]) - 1.0).v) == [3., 1, 0]
		@assert  vec((3 - ADVar(4., [1., 0])).v) == [-1., -1, 0]

	######### multiplication  ######################
		function *(X::ADVar, Y::ADVar)
			n1, m1, d1 = size(X.v)
			n2, m2, d2 = size(Y.v)

			if m1 != n2
				error("Dimensions do not match")
			elseif d1 != d2
				error("Nb of derivatives do not match")
			end

			tmp = Array(Float64, (n1, m2, d1))
			Xv = reshape(X.v[1:(n1*m1)], (n1, m1))
			Yv = reshape(Y.v[1:(n2*m2)], (n2, m2))

			tmp[1:(n1*m2)] = Xv * Yv

			# tmp = Xv * reshape(Y2[(n2*m2+1):end], (n2, m2*(d1-1)))
			# tmp += reshape(Y2[(n2*m2+1):end], (n2, m2*(d1-1))) * Yv
			i1, i2, i3 = 1, 1, 1
			for i in 2:d1
				i1 += n1*m1
				i2 += n2*m2
				i3 += n1*m2
				# println(i3:(i3+n1*m2-1)) 
				tmp[i3:(i3+n1*m2-1)] = Xv * reshape(Y.v[i2:(i2+n2*m2-1)], (n2, m2)) +
					reshape(X.v[i1:(i1+n1*m1-1)], (n1, m1)) * Yv 
			end
			# ADArray(permute(tmp, [3, 1, 2]))
			ADVar(tmp)
		end

		function *(x::ADVar, y::Any)
			if isa(y, Number)
				return(ADVar(x.v * y))
			else
				if ndims(y) > 2
					error("second argument has too many dimensions")
				elseif size(y, 1) != size(x.v, 2)
					error("Dimensions do not match")
				elseif ndims(y) == 1
					y = reshape(y, (length(y), 1))
				end

				tmp = Array(Float64, (size(x.v, 1), size(y, 2), size(x.v, 3)))
				for i in 1:size(x.v, 3) # for each value & derivative layers
					tmp[:, :, i] = x.v[:, :, i] * y
				end
				return(ADVar(tmp))
			end
		end

		function *(x::Any, y::ADVar)
			if isa(x, Number)
				return(ADVar(x * y.v))
			else
				if ndims(x) > 2
					error("first argument has too many dimensions")
				elseif size(y.v, 1) != size(x, 2)
					error("Dimensions do not match")
				elseif ndims(x) == 1
					x = reshape(x, (length(x), 1))
				end

				tmp = Array(Float64, (size(x, 1), size(y.v, 2), size(y.v, 3)))
				for i in 1:size(y.v, 3) # for each value & derivative layers
					tmp[:, :, i] = x * y.v[:, :, i]
				end
				return(ADVar(tmp))
			end
		end

		a = ADVar(4., [1., 0])
		b = ADVar(2., [0., 1])

		@assert  vec((a * b).v) == [8., 2, 4]
		@assert  vec((a * 3).v) == [12., 3, 0]

		@assert  size((a * [1 2]).v) == (1, 2, 3)
		@assert  [(a * [1 2]).v[i] for i in 1:6] == [4, 8, 1, 2, 0, 0]
		@assert  size(([1, 2] * a).v) == (2, 1, 3)
		@assert  [([1, 2] * a).v[i] for i in 1:6] == [4, 8, 1, 2, 0, 0]

	######### division  ######################
		function /(x::ADVar, y::ADVar)
			if ! isscalar(x) || ! isscalar(y)
				error("Division for arrays not implemented yet")
			end
		 	ADVar(x.v[1] / y.v[1], (x.v[2:end]*y.v[1] - y.v[2:end]*x.v[1]) / (y.v[1] * y.v[1]))
		end
		function /(x::ADVar, y::Any)
			if !isa(y, Number)
				error("Division for arrays not implemented yet")
			end
			ADVar(x.v / y)
		end
		function /(x::Any, y::ADVar)
			if !isa(x, Number)
				error("Division of arrays not implemented yet")
			elseif !isscalar(y)
				error("Division by arrays not implemented yet")
			end
			ADVar(convert(Float64, x), zeros(size(y.v, 3)-1)) / y
		end

		a = ADVar(4., [1., 0])
		b = ADVar(2., [0., 1])
isa(4., Number)
		@assert  vec((a / b).v) == [2, 0.5, -1]
		@assert  vec((a / 4).v) == [1, 0.25, 0]
		@assert  vec((4. / a).v) == [1, -0.25, 0]

	######### power  ######################
		function ^(x::ADVar, y::ADVar)
			if ! isscalar(x) || ! isscalar(y)
				error("power for arrays not implemented yet")
			end
		 	ADVar(x.v[1] ^ y.v[1], 
		 		(y.v[1] * x.v[1]^(y.v[1]-1.)) * x.v[2:end] +
		 		(log(x.v[1]) * x.v[1]^y.v[1]) * y.v[2:end])
		end
		^(x::ADVar, y::Integer) = x ^ ADVar(convert(Float64, y), zeros(size(x.v,3)-1)) 
		^(x::ADVar, y::Float64) = x ^ ADVar(y, zeros(size(x.v,3)-1)) 
		^(x::Float64, y::ADVar) = ADVar(convert(Float64, x), zeros(size(y.v,3)-1)) ^ y

		x = ADVar(3.0, [1., 0])
		y = 3.
		@assert vec((ADVar(3.0, [1., 0]) ^ 3.).v) == [27., 27, 0]
		@assert norm(vec((2.0 ^ ADVar(3., [1., 0])).v) -
		[8., 5.54518, 0]) < 1e-5
		@assert norm(vec((ADVar(3.0, [1., 0]) ^ ADVar(3.0, [0., 1])).v) -
			[27., 27, 29.6625] ) < 5e-5

	######### misc  ######################
		function log(x::ADVar)
			tmp = similar(x.v)
			tmp[:, :, 1] = log(x.v[:,:, 1])
			for i in 2:size(x.v,3)
				tmp[:, :, i] = x.v[:, :, i] ./ x.v[:,:, 1]
			end
			ADVar(tmp)
		end

		conj(x::ADVar) = x
		function ctranspose(x::ADVar)
			tmp = Array(Float64, size(x.v))
			for i in 1:size(x.v, 3)
				tmp[:,:,i] = x.v[:,:,i]'
			end
			ADVar(tmp)
		end

		@assert norm( vec(log(ADVar(4., [1., 3])).v) - [1.386294, 0.25, 0.75]) < 1e-6

end

