############################################################################
#  New type definition + method defs for automated derivation
############################################################################

module ADlib

	using Base

	import Base.+, Base.*, Base./
	import Base.log, Base.-, Base.^
	import Base.conj, Base.ctranspose, Base.transpose
	import Base.sum, Base.size, Base.ndims, Base.ref
	import Base.Vector

	# extended operator
	export ADVar
	export +, *, /, log, -, ^, conj
	export size, ndims

	type ADVar
		v::Array{Float64,3}
	end
	@assert vec(ADVar(ones(Float64, (1, 1, 4))).v) == [1, 1, 1, 1]

	isscalar(x::ADVar) = size(x.v)[1:2] == (1, 1)
	size(x::ADVar) = size(x.v)
	size(x::ADVar, d::Any) = size(x.v, d)
	@assert size(ADVar(ones(Float64, (1, 1, 4))).v) == (1, 1, 4)
	@assert size(ADVar(ones(Float64, (1, 1, 4)))) == (1, 1, 4)
	@assert size(ADVar(ones(Float64, (1, 1, 4))), 1) == 1
	@assert size(ADVar(ones(Float64, (1, 1, 4))), 3) == 4

	function ADVar{T<:Number}(x::T, dim::Integer, index::Integer)
	 	tmp = Array(Float64, (1,1,dim+1))
	 	tmp[1:numel(tmp)] = [x, [i==index ? 1.0 : 0.0 for i in 1:dim]]
	 	ADVar(tmp)
	end
	@assert vec(ADVar(1.0, 5, 2).v) == [1., 0, 1, 0, 0, 0]
	@assert vec(ADVar(1, 5, 2).v) == [1., 0, 1, 0, 0, 0]

	ADVar{T<:Number, S<:Number}(x::T, d::Vector{S}) =
	(tmp = Array(Float64, (1, 1, length(d)+1)) ; tmp[1:numel(tmp)] = [x, d[1:end]] ;ADVar(tmp))

	function ADVar{T<:Number, S<:Number}(x::T, d::Vector{S})
	 	tmp = Array(Float64, (1, 1, length(d)+1))
	 	tmp[1:numel(tmp)] = [x, d[1:end]]
	 	ADVar(tmp)
	end
	@assert vec(ADVar(2., [1., 2, 3]).v) == [2., 1, 2, 3] 
	@assert vec(ADVar(2., [1, 2, 3]).v) == [2., 1, 2, 3] 
	@assert vec(ADVar(2, [1., 2, 3]).v) == [2., 1, 2, 3] 

	function ADVar{T<:Number}(x::Array{T,1})
	 	tmp = zeros(Float64, (size(x, 1), 1, size(x, 1)+1))
	 	tmp[:, 1, 1] = reshape(x, (size(x, 1), 1, 1))
	 	for i in 1:size(x,1)
	 		tmp[i, 1, i+1] = 1.0
	 	end
	 	ADVar(tmp)
	end
	@assert reshape(ADVar([1., 2, 3]).v, 12) == [1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1] 

	# x= ADVar([1., 2, 3])
	# d = 2:3

	function ref(x::ADVar, d...)
	 	tmp = ref(x.v[:, :, 1], d)
	 	tmp2 = Array(Float64, size(tmp,1), size(tmp,2), size(x,3))
	 	tmp2[:, :, 1] = tmp
	 	for i in 2:size(x,3)
	 		tmp2[:, :, i] = ref(x.v[:, :, i], d)
	 	end
	 	ADVar(tmp2)
	end
	
	ndims(x::ADVar) = ndims(x.v) - 1

	# @assert reshape(ADVar([1., 2, 3])[2:3,:], 8) == [1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1] 


	######### addition  ######################
		+(x::ADVar, y::ADVar) = ADVar(x.v + y.v)

		function +{T<:Number}(x::ADVar, y::T)  
			for i in 1:(size(x.v,1)*size(x.v,2))
				x.v[i] += y
			end
			x
		end
		+{T<:Number}(x::T, y::ADVar) = +(y, x)

		function +{T<:Number}(x::ADVar, y::Array{T}) 
			if ndims(y) > 2
				error("second argument has too many dimensions")
			elseif ndims(y) == 1
				y = reshape(y, (length(y), 1))
			end
			if size(x.v)[1:2] != size(y)
				error("Dimensions do not match")
			end

			for i in 1:(size(x.v,1)*size(x.v,2))
				x.v[i] += y[i]
			end
			return(x)
		end
		+{T<:Number}(x::Array{T}, y::ADVar) = +(y, x)

		@assert vec((ADVar(4., [1., 0]) + ADVar(4., [0., 1])).v) == [8, 1, 1]
		@assert vec((ADVar(4., [1., 0]) + 4).v) == [8, 1, 0]
		@assert vec((3. + ADVar(3., [0., 2])).v) == [6, 0, 2]

	######### substraction ######################
		-(x::ADVar) = ADVar(-x.v)  # unary
		-(x::ADVar, y::ADVar) = ADVar(x.v - y.v)
		function -{T<:Number}(x::ADVar, y::T) 
			for i in 1:(size(x.v,1)*size(x.v,2))
				x.v[i] -= y
			end
			return(x)
		end
		-{T<:Number}(x::T, y::ADVar) = -(y - x)

		function -{T<:Number}(x::ADVar, y::Array{T}) 
				if ndims(y) > 2
					error("second argument has too many dimensions")
				elseif ndims(y) == 1
					y = reshape(y, (length(y), 1))
				end
				if size(x.v)[1:2] != size(y)
					error("Dimensions do not match")
				end

				for i in 1:(size(x.v,1)*size(x.v,2))
					x.v[i] -= y[i]
				end
				return(x)
		end
		-{T<:Number}(x::Array{T}, y::ADVar) = -(y - x)

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

			i1, i2, i3 = 1, 1, 1
			for i in 2:d1
				i1 += n1*m1
				i2 += n2*m2
				i3 += n1*m2
				tmp[i3:(i3+n1*m2-1)] = Xv * reshape(Y.v[i2:(i2+n2*m2-1)], (n2, m2)) +
					reshape(X.v[i1:(i1+n1*m1-1)], (n1, m1)) * Yv
			end
			ADVar(tmp)
		end

		*{T<:Number}(x::ADVar, y::T) = ADVar(x.v * y)
		*{T<:Number}(x::T, y::ADVar) = ADVar(x * y.v)

		function *{T<:Number}(x::ADVar, y::Array{T})
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

		function *{T<:Number}(x::Array{T}, y::ADVar)
			if ndims(x) > 2
				error("first argument has too many dimensions")
			elseif size(y.v, 1) != size(x, 2)
				error("Dimensions do not match")
			elseif ndims(x) == 1
				# if size(y.v,2) == 1 # specific shortcut for inner products
				# 	return(sum([x[i] * y.v[i] for i in 1::]))
				# end
				x = reshape(x, (length(x), 1))
			end

			tmp = Array(Float64, (size(x, 1), size(y.v, 2), size(y.v, 3)))
			for i in 1:size(y.v, 3) # for each value & derivative layers
				tmp[:, :, i] = x * y.v[:, :, i]
			end
			return(ADVar(tmp))
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

		/{T<:Number}(x::ADVar, y::T) = ADVar(x.v / y)

		function /{T<:Number}(x::T, y::ADVar)
			if !isscalar(y)
				error("Division by arrays not implemented yet")
			end
			ADVar(x, zeros(size(y.v, 3)-1)) / y
		end

		a = ADVar(4., [1., 0])
		b = ADVar(2., [0., 1])

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
		^{T<:Number}(x::ADVar, y::T) = x ^ ADVar(y, zeros(size(x.v,3)-1)) 
		^{T<:Number}(x::T, y::ADVar) = ADVar(x, zeros(size(y.v,3)-1)) ^ y

		x = ADVar(3.0, [1., 0])
		y = 3.
		@assert vec((ADVar(3.0, [1., 0]) ^ 3.).v) == [27., 27, 0]
		@assert norm(vec((2.0 ^ ADVar(3., [1., 0])).v) -
		[8., 5.54518, 0]) < 1e-5
		@assert norm(vec((ADVar(3.0, [1., 0]) ^ ADVar(3.0, [0., 1])).v) -
			[27., 27, 29.6625] ) < 5e-5

	######### misc  ######################
		function log(x::ADVar)
			for i in 2:size(x.v,3)
				x.v[:, :, i] = x.v[:, :, i] ./ x.v[:,:, 1]
			end
			x.v[:, :, 1] = log(x.v[:,:, 1])
			x
		end

		@assert norm( vec(log(ADVar(4., [1., 3])).v) - [1.386294, 0.25, 0.75]) < 1e-6

		conj(x::ADVar) = x

		function transpose(x::ADVar)
			tmp = Array(Float64, (size(x.v,2), size(x.v,1), size(x.v,3)))
			for i in 1:size(x.v, 3)
				tmp[:,:,i] = x.v[:,:,i]' #'
			end
			ADVar(tmp)
		end

		function ctranspose(x::ADVar)
			tmp = Array(Float64, (size(x.v,2), size(x.v,1), size(x.v,3)))
			for i in 1:size(x.v, 3)
				tmp[:,:,i] = x.v[:,:,i]' #'
			end
			ADVar(tmp)
		end

		function sum(x::ADVar)
			tmp = Array(Float64, (1, 1, size(x.v,3)))
			for i in 1:size(x.v, 3)
				tmp[1, 1, i] = sum(x.v[:,:,i])
			end
			ADVar(tmp)
		end



end

