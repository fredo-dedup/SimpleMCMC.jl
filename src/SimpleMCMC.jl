# module SimpleMCMC

	# import Base.*

	# export                                  # types
	#     simpleMCMC,
	#     expexp,
	#     parseParams

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
				ex = :(acc = acc + sum(logpdf($(ex.args[3]), $(ex.args[2]))))
			else 
				ex = Expr(ex.head, { parseModel(e) for e in ex.args}, Any)
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
# params
	function parseParams(ex::Expr)
		println(ex.head, " -> ")
		@assert contains([:block, :tuple], ex.head)
		index = 1
		assigns = {}
		for e in ex.args
			if e.head == :(::)
				@assert typeof(e.args[1]) == Symbol
				@assert typeof(e.args[2]) == Expr
				e2 = e.args[2]
				if e2.args[1] == :scalar
					push(assigns, :($(e.args[1]) = beta[$index]))
					index += 1
				elseif typeof(e2.args[1]) == Expr
					e3 = e2.args[1].args
					if e3[1] == :vector
						nb = eval(e3[2])
						push(assigns, :($(e.args[1]) = beta[$index:$(nb+index-1)]))
						index += nb
					end
				end
			end
		end

		(index-1, Expr(:block, assigns, Any))
	end

	###############################################################
	function simpleMCMC7(model::Expr, params::Expr, steps::Integer, scale::Real)
		# steps = 10
		# scale = 0.1
		local beta
		local __lp
		local nbeta

		model2 = parseModel(model)
		println(model2)
		(nbeta, parmap) = parseParams(params)
		println(parmap)

		beta = ones(nbeta)

		println("beta : ", beta, size(beta))

		eval(parmap)
		println("sigma :", sigma)
		println("vars : ", vars)

		__lp = 0.0
		eval(model2)
		println("__lp :", __lp)

		samples = zeros(Float64, (steps, 2+nbeta))

		loop = quote
		 	for __i in 1:steps
				oldbeta, beta = beta, beta + randn(nbeta) * scale

				$parmap # eval(parmap)
		 		old__lp, __lp = __lp, 0.0

				$model2  # eval(model2)
				if rand() > exp(__lp - old__lp)
					__lp, beta = old__lp, oldbeta
				end
				samples[__i, :] = vcat(__lp, beta)
			end
		end
		println(loop)

		eval(loop)
		samples
	end

	function simpleMCMC9(model::Expr, params::Expr, steps::Integer, scale::Real)
		# steps = 10
		# scale = 0.1
		local beta
		local nbeta

		model2 = parseModel(model)
		println(model2)
		(nbeta, parmap) = parseParams(params)
		println(parmap)

		beta = ones(nbeta)
		println("beta : ", beta, size(beta))

		eval(quote
			function loop(beta::Vector{Float64})
				local acc

				$parmap

				acc = 0.0

				$model2

				return(acc)
			end
		end)
		__lp = loop(beta)
		println("loop 1: ", __lp)

		samples = zeros(Float64, (steps, 2+nbeta))

		__lp = -Inf
	 	for __i in 1:steps
			oldbeta, beta = beta, beta + randn(nbeta) * scale

	 		old__lp, __lp = __lp, loop(beta)

			if rand() > exp(__lp - old__lp)
				__lp, beta = old__lp, oldbeta
			end
			samples[__i, :] = vcat(__lp, (old__lp != __lp), beta)
		end

		samples
	end

	function simpleMCMC10(model::Expr, params::Expr, steps::Integer, burnin::Integer)
		# (steps = 10, scale = 0.1)
		local beta, nbeta
		local jump, S, eta, SS
		const local target_alpha = 0.234

		model2 = parseModel(model)
		(nbeta, parmap) = parseParams(params)

		beta = ones(nbeta)

		eval(quote
			function __loglik(beta::Vector{Float64})
				local acc
				$parmap
				acc = 0.0
				$model2
				return(acc)
			end
		end)

		__lp = __loglik(beta)
		@assert __lp != -Inf

		samples = zeros(Float64, (steps, 2+nbeta))
		S = eye(nbeta)
	 	for i in 1:steps	 		
			jump = 0.1 * randn(nbeta)
			oldbeta, beta = beta, beta + S * jump
			# println(diag(S))

	 		old__lp, __lp = __lp, __loglik(beta)
	 		alpha = min(1, exp(__lp - old__lp))
			if rand() > exp(__lp - old__lp)
				__lp, beta = old__lp, oldbeta
			end
			samples[i, :] = vcat(__lp, (old__lp != __lp), beta)

			# eta = min(1, nbeta*i^(-2/3))
			eta = min(1, nbeta * (i <= burnin ? 1 : i-burnin)^(-2/3))
			SS = (jump * jump') / (jump' * jump)[1,1] * eta * (alpha - target_alpha)
			SS = S * (eye(nbeta) + SS) * S'
			S = chol(SS)
			S = S'
		end

		return(samples)
	end

	function simpleMCMC10(model::Expr, params::Expr, steps::Integer) 
		simpleMCMC10(model::Expr, params::Expr, steps::Integer, min(1, div(steps,2)))
	end

###################################################################################

ex = quote
	sigma::Real
	coefs::Vector(nbeta)

	
end


	function parseModel(ex)
		if typeof(ex) == Expr
			if ex.args[1]== :~
				ex = :(acc = acc + sum(logpdf($(ex.args[3]), $(ex.args[2]))))
			else 
				ex = Expr(ex.head, { parseModel(e) for e in ex.args}, Any)
			end
		end
		ex
	end



function logpdf(d::Exponential, x::Real)
    x <= 0. ? -Inf : (-x/d.scale) - log(d.scale)
end


# end



#####################################################################


	function parseExpr(ex::Expr)
		println(ex.head, " -> ")
		contains([:block, :tuple], ex.head) ? nothing : error("not a block or tuple")

		parseExpr(ex, {}, 0)
	end

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
			
			elseif e.head == :~
				e = :(acc = acc + sum(logpdf($(ex.args[3]), $(ex.args[2]))))
				
				push(block, e)
			else 
				lex = {}
				for e2 in e.args
					if typeof(e2) == Expr
						(e3, assigns, index) = parseExpr(e2, assigns, index::Integer)
					else
						e3 = e2
					end
					push(lex, e3)
				end
				push(block, Expr(e.head, lex, Any))
			end

			
		end
		println("---", block, "---")
		return(Expr(ex.head, block, Any), assigns, index)
	end




model = quote
	sigma::scalar
	vars::vector(nbeta)

	sigma ~ Gamma(2,1)
	vars ~ Normal(0,1)

	resid = Y - X * vars
	resid ~ Normal(0, sigma)
end

parseExpr(model)
expexp(model)
expexp(:(vector(nbeta)))
