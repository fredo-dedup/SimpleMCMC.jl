##########################################################################################
#
#    'derive' function containing derivation rules + extension of distributions to allow
#        vector parameters
#
##########################################################################################

# TODO : include target size info
# TODO : include input size info
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
				:(isa($a1, Real) ? dot([$ds], [$a2]) : $ds * $a2')
			else
				:(isa($a2, Real) ? dot([$ds], [$a1]) : $a1' * $ds)
			end
			# index == 1 ? :($ds * $a2') : :($a1' * $ds)

		elseif op == :.*  
			if index == 1 
				:(isa($a1, Real) ? dot([$ds], [$a2]) : $ds .* $a2)
			else
				:(isa($a2, Real) ? dot([$ds], [$a1]) : $ds .* $a1)
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

		elseif op == :/ # Note that this will not work if both args are arrays but without warning
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

#TODO : implement simplifications here ?  (eg. logpdf(Normal)) as this is not always done in Distributions


for d in [:Normal, :Weibull, :Uniform]
	fsym = symbol("logpdf$d")

	@eval ($fsym)(a::Real, b::Real, x::Real) = sum([logpdf(($d)(a, b), x)])

	@eval ($fsym)(a::Real, b::Real, x::Vector) = sum([logpdf(($d)(a, b), x)])

	eval(quote
		function ($fsym)(a::Union(Real, Vector), b::Union(Real, Vector), x::Union(Real, Vector))
			res = 0.0
			for i in 1:max(length(a), length(b), length(x))
				res += logpdf(($d)(next(a,i)[1], next(b,i)[1]), next(x,i)[1])
			end
			res
		end
	end) 
end



logpdfTestDiff(x) = sum([x])  # dummy distrib for testing

