
module SimpleMCMC

	# using Base
	# using String

	export simpleRWM, expexp, parseExpr

	####################################################################################

	function expexp(ex::Expr, ident...)
		ident = (length(ident)==0 ? 0 : ident[1])::Integer
		println(rpad("", ident, " "), ex.head, " -> ")
		for e in ex.args
			typeof(e)==Expr ? expexp(e, ident+3) : println(rpad("", ident+3, " "), "[", typeof(e), "] : ", e)
		end
	end

#####################################################################

	function simpleRWM(model::Expr, steps::Integer, burnin::Integer, init::Any)
		local beta, nparams
		local jump, S, eta, SS
		const local target_alpha = 0.234

		(model2, param_map, nparams) = parseExpr(model)

		if typeof(init) == Array{Float64,1}
			numel(init) != nparams ? error("$nparams initial values expected, got $(numel(init))") : nothing
			beta = init
		elseif typeof(init) <: Real
			beta = [ convert(Float64, init)::Float64 for i in 1:nparams]
		else
			error("cannot assign initial values (should be a Real or vector of Reals)")
		end

		eval(quote
			function __loglik(beta::Vector{Float64})
				local acc
				$(Expr(:block, param_map, Any))
				acc = 0.0
				$model2
				return(acc)
			end
		end)

		__lp = Main.__loglik(beta)
		__lp == -Inf ? error("Initial values out of model support, try other values") : nothing

		samples = zeros(Float64, (steps, 2+nparams))
		S = eye(nparams)
	 	for i in 1:steps	 # i=1; burnin=10		
	 		# print(i, " old beta = ", round(beta[1],3))
			jump = 0.1 * randn(nparams)
			oldbeta, beta = beta, beta + S * jump
			# print("new beta = ", round(beta[1], 3), " diag = ", round(diag(S), 3))

	 		old__lp, __lp = __lp, Main.__loglik(beta)
	 		# println(" lp= ", round(__lp, 3))

	 		alpha = min(1, exp(__lp - old__lp))
			if rand() > exp(__lp - old__lp)
				__lp, beta = old__lp, oldbeta
			end
			samples[i, :] = vcat(__lp, (old__lp != __lp), beta)

			eta = min(1, nparams*i^(-2/3))
			# eta = min(1, nparams * (i <= burnin ? 1 : i-burnin)^(-2/3))
			SS = (jump * jump') / (jump' * jump)[1,1] * eta * (alpha - target_alpha)
			SS = S * (eye(nparams) + SS) * S'
			S = chol(SS)
			S = S'
		end

		return(samples)
	end

	simpleRWM(model::Expr, steps::Integer) = simpleRWM(model, steps, min(1, div(steps,2)))
	simpleRWM(model::Expr, steps::Integer, burnin::Integer) = simpleRWM(model, steps, burnin, 1.0)

	###################################################################################

# function logpdf(d::Exponential, x::Real)
#     x <= 0. ? -Inf : (-x/d.scale) - log(d.scale)
# end


# # end



	#####################################################################

	function parseExpr(ex::Expr, assigns::Array{Any}, index::Integer)
		# ex = model ; assigns = {} ; index = 1
		block = {}
		for e in ex.args  # e = ex.args[4]
			if !isa(e, Expr)
				push(block, e)

			elseif e.head == :(::)  #  model param declaration
				typeof(e.args[1]) == Symbol ? nothing : error("not a symbol on LHS of ~ : $(e.args[1])")
				par = e.args[1]  # param symbol defined here

				if e.args[2] == :scalar  #  simple decl : var::scalar
					push(assigns, :($(par) = beta[$(index+1)]))
					index += 1

				elseif typeof(e.args[2]) == Expr && e.args[2].head == :call
					e2 = e.args[2].args
					if e2[1] == :vector
						nb = eval(e2[2])
						if !isa(nb, Integer) || nb < 1 
							error("invalid size $(e2[2]) = $(nb)")
						end
						push(assigns, :($(par) = beta[$(index+1):$(nb+index)]))
						index += nb
					else
						error("unknown parameter type $(e2[1])")
					end
				else
					error("unknown parameter expression $(e)")
				end
			
			elseif e.head == :call && e.args[1] == :~
				e = :(acc = acc + sum(logpdf($(e.args[3]), $(e.args[2]))))
				# println("++++", e)
				push(block, e)
			else 
				lex = {}
				for e2 in e.args
					if typeof(e2) == Expr
						(e3, assigns, index) = parseExpr(e2, assigns, index)
					else
						e3 = e2
					end
					push(lex, e3)
				end
				push(block, Expr(e.head, lex, Any))
			end
			
		end
		# println("---", block, "---")
		return(Expr(ex.head, block, Any), assigns, index)
	end

	parseExpr(ex::Expr) = parseExpr(ex, {}, 0) # initial function call

end  # end of module