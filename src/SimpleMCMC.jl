module SimpleMCMC

	import Base.*

	export                                  # types
	    simpleMCMC,
	    expexp

	####################################################################################

	function expexp(ex::Expr, ident...)
		ident = (length(ident)==0 ? 0 : ident[1])::Integer
		println(rpad("", ident, " "), ex.head, " -> ")
		for e in ex.args
			typeof(e)==Expr ? expexp(e, ident+3) : println(rpad("", ident+3, " "), "[", typeof(e), "] : ", e)
		end
	end

	####################################################################

	function parseModel(ex)
		if typeof(ex) == Expr
			if ex.args[1]== :~
				ex = :(__lp = __lp + sum(logpdf($(ex.args[3]), $(ex.args[2]))))
			else 
				ex = Expr(ex.head, { mcmc(e) for e in ex.args}, Any)
			end
		end
		ex
	end

	# function mlog_normal(x::Float64, sigma::Float64) 
	# 	tmp = x / sigma
	# 	-tmp*tmp 
	# end
	# function mlog_normal(x::Vector{Float64}, sigma::Float64) 
	# 	tmp = x / sigma
	# 	- sum(tmp .* tmp)
	# end

	###############################################################

	function parseParams(ex)
		println(ex.head, " -> ")
		@assert params.head == :block
		index1 = 1
		index2 = 1
		assigns = {}
		for e in ex.args
			if e.head == :(::)
				@assert typeof(e.args[1]) == Symbol
				@assert typeof(e.args[2]) == Expr
				e2 = e.args[2]
				if e2.args[1] == :scalar
					push(assigns, :($(e.args[1]) = beta[$index1]))
					index1 += 1
				elseif typeof(e2.args[1]) == Expr
					e3 = e2.args[1].args
					if e3[1] == :vector
						nb = eval(e3[2])
						push(assigns, :($(e.args[1]) = beta[$index1:$(nb+index1-1)]))
						index1 += nb
					end
				end
			end
		end

		Expr(:block, assigns, Any)
	end

	###############################################################
	function simpleMCMC(model::Expr, params::Expr, step::Integer, scale::Real)

		model2 = parseModel(model)
		parmap = parseParams(params)

		loop = quote 
		 	for __i in 1:steps
				oldbeta, beta = beta, beta + randn(1+nbeta) * scale

				$parmap
				# eval(setmap)
		 		old__lp, __lp = __lp, 0.0

				$model2
				# eval(modelExpr)
				if rand() > exp(__lp - old__lp)
					__lp, beta = old__lp, oldbeta
				end
				samples[__i, :] = vcat(__lp, beta)
			end
		end

		samples = zeros(Float64, (steps, 2+nbeta))

		eval(loop)
	end

end
