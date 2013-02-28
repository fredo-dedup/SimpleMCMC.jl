##########################################################################################
#
#    function 'derive' returning the expr for the gradient calculation
#    +  definition of functions logpdf... 
#
##########################################################################################

# TODO : add operators :' , :vcat, max and min ?

function derive(opex::Expr, index::Integer, dsym::Union(Expr,Symbol))
	op = opex.args[1]  # operator
	vs = opex.args[1+index]
	ds = symbol("$(DERIV_PREFIX)$dsym")
	args = opex.args[2:end]

	length(args) >= 1 ? a1=args[1] : nothing
	length(args) >= 2 ? a2=args[2] : nothing
	length(args) >= 3 ? a3=args[3] : nothing
	length(args) >= 4 ? a4=args[4] : nothing
	# TODO : turn into a loop

	dexp =  
		if op == :+  
			:(isa($vs, Real) ? sum($ds) : $ds)

		elseif op == :sum 
			ds

		elseif op == :log
			:($ds ./ $vs)

		elseif op == :sin
			:(cos($vs) .* $ds)

		elseif op == :cos
			:(-sin($vs) .* $ds)

		elseif op == :exp
			:(exp($vs) .* $ds)

		elseif op == :- && length(args) == 1
			:(-$ds)

		elseif op == :- && length(args) == 2
			if index == 1 
				:(isa($a1, Real) ? sum($ds) : $ds)
			else
				:(isa($a2, Real) ? -sum($ds) : -$ds)
			end

		elseif op == :*
			if index == 1 
				:(isa($a1, Real) ? sum([$ds .* $a2]) : $ds * $a2')
			else
				:(isa($a2, Real) ? sum([$ds .* $a1]) : $a1' * $ds)
			end
			# index == 1 ? :($ds * $a2') : :($a1' * $ds)

		elseif op == :.*  
			if index == 1 
				:(isa($a1, Real) ? sum([$ds .* $a2]) : $ds .* $a2)
			else
				:(isa($a2, Real) ? sum([$ds .* $a1]) : $ds .* $a1)
			end
			# index == 1 ? :(sum($a2) .* $ds) : :(sum($a1) .* $ds)

		elseif op == :^ # Both args reals
			index == 1 ? :($a2 * $a1 ^ ($a2-1) * $ds) : :(log($a1) * $a1 ^ $a2 * $ds)

		elseif op == :.^
			if index == 1 
				:(isa($a1, Real) ? sum([$a2 .* $a1 .^ ($a2-1) .* $ds]) : $a2 .* $a1 .^ ($a2-1) .* $ds)
			else
				:(isa($a2, Real) ? sum([log($a1) .* $a1 .^ $a2 .* $ds]) : log($a1) .* $a1 .^ $a2 .* $ds)
			end

		elseif op == :/ # FIXME : this will not work if both args are arrays but without warning
			if index == 1 
				:(isa($a1, Real) ? sum([$ds ./ $a2]) : $ds ./ $a2 )
			else
				:(isa($a2, Real) ? sum([- $a1 ./ ($a2 .* $a2) .* $ds]) : - $a1 ./ ($a2 .* $a2) .* $ds )
			end

		elseif op == :./
			if index == 1 
				:(isa($a1, Real) ? sum([$ds ./ $a2]) : $ds ./ $a2 )
			else
				:(isa($a2, Real) ? sum([- $a1 ./ ($a2 .* $a2) .* $ds]) : - $a1 ./ ($a2 .* $a2) .* $ds )
			end

		elseif op == :dot
			index == 1 ? :($a2 * $ds) : :($a1 * $ds)

		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfNormal)) 
			if index == 1 # mu
				:( (tmp = [($a3 - $a1 ) ./ ($a2 .^ 2)] * $ds ; isa($a1, Real) ? sum(tmp) : tmp) * $ds)
			elseif index == 2 # sigma
				:( (tmp = (($a3 - $a1).^2 ./ $a2.^2 - 1.0) ./ $a2 * $ds ; isa($a2, Real) ? sum(tmp) : tmp) * $ds)
			else # x  
				:( (tmp = [($a1 - $a3 ) ./ ($a2 .^ 2)] * $ds ; isa($a3, Real) ? sum(tmp) : tmp) * $ds)
			end
		
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfUniform)) 
			if index == 1 # a   
				:( (tmp = [$a1 .<= $a3 .<= $a2] .* ($ds ./ ($a2 - $a1)) ; isa($a1, Real) ? sum([tmp]) : tmp) .* $ds)
			elseif index == 2 # b
			 	:( (tmp = [$a1 .<= $a3 .<= $a2] .* (-$ds ./ ($a2 - $a1)) ; isa($a2, Real) ? sum([tmp]) : tmp) .* $ds)
			else # x  
			 	:( 0.0 )
			end
		
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfWeibull)) 
			if index == 1 # shape
				:( (tmp = (1.0 - ($a3./$a2).^$a1) .* log($a3./$a2) + 1./$a1 * $ds ; isa($a1, Real) ? sum([tmp]) : tmp) .* $ds)
			elseif index == 2 # scale
			 	:( (tmp = (($a3./$a2).^$a1 - 1.0) .* $a1 ./ $a2 * $ds ; isa($a2, Real) ? sum([tmp]) : tmp) .* $ds)
			else # x  
			 	:( (tmp = ( (1.0 - ($a3./$a2).^$a1) .* $a1 -1.0) ./ $a3 * $ds ; isa($a3, Real) ? sum([tmp]) : tmp) .* $ds)
			end
		
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfBernoulli)) 
			if index == 1 # probability
				:( (tmp = 1. ./ ($a1 - (1. - $a2)) ; isa($a1, Real) ? sum([tmp]) : tmp) .* $ds)
			else # x, x is discrete for Bernoulli therefore no derivative should be calculated
			 	error("[derive] in $opex : the gradient cannot depend on a discrete variable")
			end
		
		#  fake distribution to test gradient code
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfTestDiff)) 
			ds
		
		else
			error("[derive] Doesn't know how to derive operator $op")
		end


	return :($(symbol("$(DERIV_PREFIX)$vs")) += $dexp )
end



###############  hooks into Distributions library  ###################
#  (allows to vectorize on distributions parameters)

#TODO ? : implement here functions that can be simplified (eg. logpdf(Normal)) 
# as this is not always done in Distributions

#  1 parameter distributions
# for d in [:Bernoulli]
# 	fsym = symbol("logpdf$d")  # function to be created
#     dlf = eval(d) <: DiscreteDistribution ? :logpmf : :logpdf # function to be called depends on distribution type

# 	@eval ($fsym)(a::Real, x::Real) = ($dlf)(($d)(a), x)

# 	@eval ($fsym)(a::Real, x::Array) = sum([($dlf)(($d)(a), x)])

# 	eval(quote
# 		function ($fsym)(a::Union(Real, Array), x::Union(Real, Array))
# 			res = 0.0
# 			for i in 1:max(length(a), length(x))
# 				res += ($dlf)(($d)(next(a,i)[1]), next(x,i)[1])
# 			end
# 			res
# 		end
# 	end) 
# end

# #  2 parameters distributions
# for d in [:Normal, :Weibull, :Uniform]
# # for d in [:Weibull, :Uniform]
# 	fsym = symbol("logpdf$d")
#     dlf = eval(d) <: DiscreteDistribution ? :logpmf : :logpdf # function to be called depends on distribution type

# 	@eval ($fsym)(a::Real, b::Real, x::Real) = ($dlf)(($d)(a, b), x)

# 	@eval ($fsym)(a::Real, b::Real, x::Array) = sum([($dlf)(($d)(a, b), x)])

# 	eval(quote
# 		function ($fsym)(a::Union(Real, Array), b::Union(Real, Array), x::Union(Real, Array))
# 			res = 0.0
# 			for i in 1:max(length(a), length(b), length(x))
# 				res += ($dlf)(($d)(next(a,i)[1], next(b,i)[1]), next(x,i)[1])
# 			end
# 			res
# 		end
# 	end) 
# end

########### distributions using libRmath ######### 
_jl_libRmath = dlopen("libRmath")

# TODO : use the signature length parameter for real !
for d in {(:Normal,  	"dnorm4",	2),
		  (:Weibull, 	"dweibull", 2),
		  (:Uniform, 	"dunif", 	2)}

	sig = tuple([Float64 for i in 1:(d[3]+1)]..., Int32)
	fsym = symbol("logpdf$(d[1])")

	eval(quote

		function ($fsym)(a::Real, b::Real, x::Real)
			local res
			res = ccall(dlsym(_jl_libRmath, $(d[2])), Float64, 
				(Float64, Float64, Float64, Int32), 
					x, a, b, 1)
			isfinite(res) ? res : throw("break loglik")
		end

		function ($fsym)(a::Union(Real, AbstractArray), 
			             b::Union(Real, AbstractArray), 
			             x::Union(Real, AbstractArray))
			local res, acc

			acc = 0.0
			for i in 1:max(length(a), length(b), length(x))
				res = ccall(dlsym(_jl_libRmath, $(d[2]) ), Float64, (Float64, Float64, Float64, Int32), 
					next(x,i)[1], next(a,i)[1], next(b,i)[1], 1)
				acc += isfinite(res) ? res : throw("break loglik")
			end
			acc
		end

	end) 

end

########## locally defined distributions #############

logpdfBernoulli(prob::Real, x::Real) = x == 0 ? log(1. - prob) : (x == 1 ? log(prob) : throw("break loglik"))

for d in [:Bernoulli]
	fsym = symbol("logpdf$d")

	eval(quote
		function ($fsym)(a::Union(Real, AbstractArray), x::Union(Real, AbstractArray))
			res = 0.0
			for i in 1:max(length(a), length(x))
				res += ($fsym)(next(a,i)[1], next(x,i)[1])
			end
			res
		end
	end) 
end

############# dummy distrib for testing ############

logpdfTestDiff(x) = sum([x]) 

