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
			index == 1 ? :($ds * $a2') : :($a1' * $ds)

		elseif op == :^
			index == 1 ? :($a2 * $vs ^ ($a2-1) * $ds) : :(log($a1) * $a1 ^ $vs * $ds)

		elseif op == :/
			index == 1 ? :($ds ./ $a2) : :(- $a1 ./ ($vs .* $vs) .* $ds)

		elseif op == :dot
			index == 1 ? :(sum($a2) .* $ds) : :(sum($a1) .* $ds)

		elseif op == :.*  # TODO : check this
			index == 1 ? :(sum($a2) .* $ds) : :(sum($a1) .* $ds)

		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfNormal)) 
			if index == 1 # mu
				:( (tmp = [($a3 - $a1 ) ./ ($a2 .^ 2)] * $ds ; length($a1)==1 ? sum(tmp) : tmp) )
			elseif index == 2 # sigma
				:( (tmp = (($a3 - $a1).^2 ./ $a2^2 - 1.0) / $a2 * $ds ; length($a2)==1 ? sum(tmp) : tmp) )
			else # x  
				:( (tmp = [($a1 - $a3 ) ./ ($a2 .^ 2)] * $ds ; length($a3)==1 ? sum(tmp) : tmp) )
			end
		
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfUniform)) 
			if index == 1 # a   
				# :( (tmp = [$a1 .<= $a3 .<= $a2] .* [$ds ./ ($a2 - $a1) .* ones(length($a3))] ; length($a1)==1 ? sum([tmp]) : tmp) )
				:( (tmp = [$a1 .<= $a3 .<= $a2] .* ($ds ./ ($a2 - $a1)) ; length($a1)==1 ? sum([tmp]) : tmp) )
			elseif index == 2 # b
			 	# :( (tmp = [$a1 .<= $a3 .<= $a2] .* [-$ds ./ ($a2 - $a1) .* ones(length($a3))] ; length($a2)==1 ? sum([tmp]) : tmp) )
			 	:( (tmp = [$a1 .<= $a3 .<= $a2] .* (-$ds ./ ($a2 - $a1)) ; length($a2)==1 ? sum([tmp]) : tmp) )
			else # x  
			 	:( 0.0 )
			end
		
		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfWeibull)) 
			if index == 1 # shape
				:( (tmp = (1.0 - ($a3./$a2).^$a1) .* log($a3./$a2) + 1./$a1 * $ds ; length($a1)==1 ? sum([tmp]) : tmp) )
			elseif index == 2 # scale
			 	:( (tmp = (($a3./$a2).^$a1 - 1.0) .* $a1 ./ $a2 * $ds ; length($a2)==1 ? sum([tmp]) : tmp) )
			else # x  
			 	:( (tmp = ( (1.0 - ($a3./$a2).^$a1) .* $a1 -1.0) ./ $a3 * $ds ; length($a3)==1 ? sum([tmp]) : tmp) )
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

#TODO ? : implement here functions that can be simplified (eg. logpdf(Normal)) as this is not always done in Distributions

logpdfNormal(mu::Real, sigma::Real, x::Real) = 			logpdf(Normal(mu, sigma), x)
logpdfNormal(mu::Real, sigma::Real, x::Vector) = 	sum([ logpdf(Normal(mu, sigma), x) ]) 

function logpdfNormal(	mu::Union(Real, Vector), 
						sigma::Union(Real, Vector), 
						x::Union(Real, Vector)) 

	res = 0.0
	for i in 1:max(length(mu), length(sigma), length(x))
		res += logpdfNormal(next(mu,i)[1], next(sigma,i)[1], next(x,i)[1])
	end
	res
end


logpdfUniform(a::Real, b::Real, x::Real) = 	logpdf(Uniform(a, b), x)
logpdfUniform(a::Real, b::Real, x::Vector) = 	sum([ logpdf(Uniform(a, b), x) ]) 

function logpdfUniform(	a::Union(Real, Vector), 
						b::Union(Real, Vector), 
						x::Union(Real, Vector)) 

	res = 0.0
	for i in 1:max(length(a), length(b), length(x))
		res += logpdf(Uniform(next(a,i)[1], next(b,i)[1]), next(x,i)[1])
	end
	res
end

logpdfWeibull(shape::Real, scale::Real, x::Real) = 		logpdf(Weibull(shape, scale), x)
logpdfWeibull(shape::Real, scale::Real, x::Vector) = 	sum([ logpdf(Weibull(shape, scale), x) ]) 

function logpdfWeibull(	shape::Union(Real, Vector), 
						scale::Union(Real, Vector), 
						x::Union(Real, Vector)) 

	res = 0.0
	for i in 1:max(length(shape), length(scale), length(x))
		res += logpdf(Weibull(next(shape,i)[1], next(scale,i)[1]), next(x,i)[1])
	end
	res
end




# logpdfUniform(a, b, x) = 			sum([ logpdf(Uniform(a, b), x) ])
# logpdfNormal(mu, sigma, x) = 		sum([ logpdf(Normal(mu, sigma), x) ])  # brackets to manage case where logpdf=-Inf
# logpdfWeibull(shape, scale, x) = 	sum([ logpdf(Weibull(shape, scale), x) ])


logpdfTestDiff(x) = x  # dummy distrib for testing

