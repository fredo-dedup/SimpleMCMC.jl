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
	args = opex.args[2:end]
	dvs = symbol("$(DERIV_PREFIX)$vs")
	ds = symbol("$(DERIV_PREFIX)$dsym")

	dexp =  
		if op == :+
			ds

		if op == :sum # TODO : check this
			????

		if op == :log
			:($ds ./ $vs)

		if op == :sin
			:(cos($vs) .* $ds)

		if op == :cos
			:(-sin($vs) .* $ds)

		if op == :exp
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
			index == 1 ? :($vs ./ $a2 .* $ds) : :(- $a1 ./ ($vs .* $vs) .* $ds)

		elseif op == :dot
			index == 1 ? :(sum($a2) .* $ds) : :(sum($a1) .* $ds)

		elseif op == :.*  # TODO : check this
			index == 1 ? :(sum($a2) .* $ds) : :(sum($a1) .* $ds)

		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfNormal)) #TODO : error
			if index == 1 # mu
				:(sum($a3 - $a1 ) / $a2),
			elseif index == 2 # sigma
				:(sum( ($a3 - $a1).^2 ./ $a2^2 - 1.0) / $a2)
			else # x  
				:(($a1 - $a3 ) ./ $a2)
			end
		
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfUniform)) #TODO : error
			if index == 1 # a   # TODO ( ? : ) vectorized ??
				:(sum( log( ($a1 .<= $a3 .<= $a2 ? 1.0 : 0.0) ./ (($a2 - $a1).^2.0) ))), 
			elseif index == 2 # b
			 	:(sum( log( -($a1 .<= $a3 .<= $a2 ? 1.0 : 0.0) ./ (($a2 - $a1).^2.0) ) )),
			else # x  
			 	:(sum( log( ($a1 .<= $a3 .<= $a2 ? 1.0 : 0.0) ./ ($a2 - $a1) ) ))
			end
		
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfWeibull)) #TODO : error
			if index == 1 # shape
				:(sum( (1.0 - ($a3./$a2).^$a1) .* log($a3./$a2) + 1./$a1)), 
			elseif index == 2 # scale
			 	:(sum( (($a3./$a2).^$a1 - 1.0) .* $a1 ./ $a2)),
			else # x  
			 	:(sum( ( (1.0 - ($a3./$a2).^$a1) .* $a1 -1.0) ./ $a3))
			end
		
		else
			error("[derive] Doesn't know how to derive binary operator $op")
		end


	return :($dvs += $dexp )
end

