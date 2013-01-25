##########################################################################################
#
#    Derivation rules and derive function definition
#
##########################################################################################

# TODO : include target size info
# TODO : include input size info

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
		if op == :+  # ERROR : false if length(vs) = 1 and length(ds) > 1
			ds

		elseif op == :sum # ERROR : false if length(vs) 
			:($ds ./ length($vs))

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
			index == 1 ? ds : :(-$ds)

		elseif op == :*
			index == 1 ? :($ds * transpose($a2)) : :(transpose($a1) * $ds)

		elseif op == :^
			index == 1 ? :($a2 * $vs ^ ($a2-1) * $ds) : :(log($a1) * $a1 ^ $vs * $ds)

		elseif op == :/
			index == 1 ? :($ds ./ $a2) : :(- $a1 ./ ($vs .* $vs) .* $ds)

		elseif op == :dot
			index == 1 ? :(sum($a2) .* $ds) : :(sum($a1) .* $ds)

		elseif op == :.*  # TODO : check this
			index == 1 ? :(sum($a2) .* $ds) : :(sum($a1) .* $ds)

		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfNormal)) #TODO : error
			if index == 1 # mu
				:( ($a3 - $a1 ) ./ $a2 )
			elseif index == 2 # sigma
				:( (($a3 - $a1).^2 ./ $a2^2 - 1.0) / $a2)
			else # x  
				:( ($a1 - $a3 ) ./ $a2 )
			end
		
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfUniform)) #TODO : error
			if index == 1 # a   # TODO ( ? : ) vectorized ??
				:( dot(([$a1 .<= $a3 .<= $a2] .* 1.0), [1 ./ ($a2 - $a1)]) )
			elseif index == 2 # b
			 	:( dot(([$a1 .<= $a3 .<= $a2] .* 1.0), [-1 ./ ($a2 - $a1)]) )
			else # x  
			 	:(0.0)
			end
		
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfWeibull)) #TODO : error
			if index == 1 # shape
				:(sum( (1.0 - ($a3./$a2).^$a1) .* log($a3./$a2) + 1./$a1))
			elseif index == 2 # scale
			 	:(sum( (($a3./$a2).^$a1 - 1.0) .* $a1 ./ $a2))
			else # x  
			 	:(sum( ( (1.0 - ($a3./$a2).^$a1) .* $a1 -1.0) ./ $a3))
			end
		
		#  fake distribution to test gradient code
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfTestDiff)) 
			ds
		
		else
			error("[derive] Doesn't know how to derive operator $op")
		end


	return :($(symbol("$(DERIV_PREFIX)$vs")) += $dexp )
end

