##########################################################################################
#
#    Derivation rules and derive function definition
#
##########################################################################################

# TODO : include target size info
# TODO : include input size info

abstract sumop
abstract plusop
abstract minusop

const tmap = {
	:- => Minusop,
	:+ => Plusop,
	:
}
diff(e::Sumop) = $ds
diff(e::Prodop) = index == 1 ? :($ds * transpose($a2)) : :(transpose($a1) * $ds)
diff(e::Dotop) = index == 1 ? :(sum($a2) * $ds) : :(sum($a1) * $ds)

:($prepare(e)) +=*  => {:($ds * transpose($(args[2]))), 
					:(transpose($(args[1])) * $ds)},

function derive(opex::Expr, index::Integer, dsym::Union(Expr,Symbol))
	op = opex.args[1]  # operator
	vs = opex.args[1+index]
	args = opex.args[2:end]
	dvs = symbol("$(DERIV_PREFIX)$vs")
	ds = symbol("$(DERIV_PREFIX)$dsym")

	# TODO : all the dict expressions are evaluated, should be deferred
	# TODO : cleanup, factorize this mess
	if length(args) == 1 # unary operators
		drules_unary = {
			:- => :(-$ds),
			:log => :($ds ./ $vs),
			:sum => ds,
			:sin => :(cos($vs) .* $ds),
			:exp => :(exp($vs) .* $ds)
		}

		assert(has(drules_unary, op), "[derive] Doesn't know how to derive unary operator $op")
		return :($dvs += $(drules_unary[op]) )

	elseif length(args) == 2 # binary operators
		drules_binary = {
			:-  => {ds, :(-$ds)},
			:+  => {ds, ds},
			:sum  => {ds, ds},
			:*  => {:($ds * transpose($(args[2]))), 
					:(transpose($(args[1])) * $ds)},
			:(.*)  => {:(sum($(args[2])) * $ds), :(sum($(args[1])) * $ds)},
			:dot => {:(sum($(args[2])) * $ds), 
					 :(sum($(args[1])) * $ds)},
			:^ => {:($(args[2]) * $vs ^ ($(args[2])-1) * $ds),
				   :(log($(args[1])) * $(args[1]) ^ $vs * $ds)},
			:/ => {:($vs ./ $(args[2]) .* $ds),
				   :(- $(args[1]) ./ ($vs .* $vs) .* $ds)}
		}

		assert(has(drules_binary, op), "[derive] Doesn't know how to derive binary operator $op")
		return :($dvs += $(drules_binary[op][index]) )

	elseif length(args) == 3 # ternary operators
		drules_ternary = {
			:sum  => {ds, ds, ds} #,
		}

		if isa(op, Symbol)
			assert(has(drules_ternary, op), "[derive] Doesn't know how to derive ternary operator $op")
			return :($dvs += $(drules_ternary[op][index]) )

		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfNormal))
			rules = {	# mu
						:(sum($(args[3]) - $(args[1]) ) / $(args[2])),
					  	# sigma
					  	:(sum( ($(args[3]) - $(args[1])).^2 ./ $(args[2])^2 - 1.0) / $(args[2])),
					  	# x
					  	:(($(args[1]) - $(args[3]) ) ./ $(args[2]))
					}
			return :($dvs += $(rules[index]) .* $ds)

		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfUniform))
			rules = {	# a
						:(sum( log( ($(args[1]) <= $(args[3]) <= $(args[2]) ? 1.0 : 0.0) ./ (($(args[2]) - $(args[1])).^2.0) ))), 
						# b
					 	:(sum( log( -($(args[1]) <= $(args[3]) <= $(args[2]) ? 1.0 : 0.0) ./ (($(args[2]) - $(args[1])).^2.0) ) )),
					 	# x
					 	:(sum( log( ($(args[1]) <= $(args[3]) <= $(args[2]) ? 1.0 : 0.0) ./ ($(args[2]) - $(args[1])) ) ))
					}
			return :($dvs += $(rules[index]) .* $ds)

		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfWeibull))
			rules = {	# shape
						:(sum( (1.0 - ($(args[3])./$(args[2])).^$(args[1])) .* log($(args[3])./$(args[2])) + 1./$(args[1]))), 
						# scale
					 	:(sum( (($(args[3])./$(args[2])).^$(args[1]) - 1.0) .* $(args[1]) ./ $(args[2]))),
					 	# x
					 	:(sum( ( (1.0 - ($(args[3])./$(args[2])).^$(args[1])) .* $(args[1]) -1.0) ./ $(args[3])))
					}
			return :($dvs += $(rules[index]) .* $ds)

		else
			error("[derive] Doesn't know how to derive ternary operator $op")	
		end

	else
		error("[derive] Doesn't know how to derive n-ary operator $op")
	end
end

