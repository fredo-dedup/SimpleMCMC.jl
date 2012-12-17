module SimpleMCMC

	using Base 

	export simpleRWM, expexp, parseExpr
	export unfoldBlock,
		localVars,
		backwardSweep,
		derive,
		unfoldExpr,
		nameFactory,
		nameTask

##########  unfoldBlock ##############

	function unfoldBlock(ex::Expr)
		assert(ex.head == :block, "[unfoldBlock] not a block")

		lb = {}  # will store expressions
		vars = {} # will store variables in the block
		for e in ex.args  #  e  = ex.args[8]
			if e.head == :line   # line number marker, no treatment
				push(lb, e)
			elseif e.head == :(=)  # assigment
				lhs = e.args[1]
				if typeof(lhs) == Symbol # simple var case
					push(vars, lhs)
				elseif typeof(lhs) == Expr && lhs.head == :ref  # vars with []
					push(vars, lhs.args[1])
				else
					error("[unfoldBlock] not a symbol on LHS of assigment $(e)") 
				end

				rhs = e.args[2]
				if typeof(rhs) == Symbol ||  (typeof(rhs) == Expr && rhs.head == :ref)
					push(lb, e)
				elseif typeof(rhs) == Expr
					ue = unfoldExpr(rhs)
					if length(ue)==1 # simple expression, no variable creation necessary
						push(lb, e)
					else # several nested expressions
						for a in ue[1:end-1]
							push(lb, a)
							lhs2 = a.args[1]
							if typeof(lhs2) == Symbol # simple var case
								push(vars, lhs2)
							elseif typeof(lhs2) == Expr && lhs2.head == :ref  # vars with []
								push(vars, lhs2.args[1])
							else
								error("[unfoldBlock] not a symbol on LHS of assigment $(e)") 
							end
						end
						push(lb, :($lhs = $(ue[end])))
					end
				else  # unmanaged kind of lhs
				 	error("[unfoldBlock] can't parse $e")
				end

			elseif e.head == :for  # TODO parse loops
				error("[unfoldBlock] can't handle $(e.head) expressions")  # TODO

			elseif e.head == :if  # TODO parse if structures
				error("[unfoldBlock] can't handle $(e.head) expressions")  # TODO

			elseif e.head == :block
				(e2, nv) = unfoldBlock(e)
				map(x->push(lb,x), e2)  # exprs are inserted in parent block without begin - end
				map(x->push(vars,x), nv)
			else
				error("[unfoldBlock] can't handle $(e.head) expressions")
			end
		end

		(Expr(:block, lb, Any), vars)
	end

	function localVars(ex::Expr)
		assert(ex.head == :block, "[localVars] not a block")

		lb = {}
		for e in ex.args  #  e  = ex.args[2]
			if e.head == :(=)  # assigment
				assert(typeof(e.args[1]) == Symbol, "[localVars] not a symbol on LHS of assigment $(e)") # TODO : manage symbol[]
				
				contains(lb, e.args[1]) ? nothing : push(lb, e.args[1])

			elseif e.head == :line  # TODO parse loops

			elseif e.head == :for  # TODO parse loops

			elseif e.head == :while  # TODO parse loops

			elseif e.head == :if  # TODO parse if structures

			elseif e.head == :block
				e2 = localVars(e)
				for v in e2
					push(lb, v)
				end
			else
				error("[localVars] can't handle $(e.head) expressions")
			end
		end

		lb
	end

##########  backwardSweep ##############

	function backwardSweep(ex::Expr, locals::Vector)
		assert(ex.head == :block, "[backwardSweep] not a block")

		lb = {}
		for e in ex.args  #  e  = ex.args[2]
			if e.head == :line   # line number marker, no treatment
				push(lb, e)
			elseif e.head == :(=)  # assigment
				lhs = e.args[1]
				if typeof(lhs) == Symbol # simple var case
					dsym = lhs
					dsymref = ""
				elseif typeof(lhs) == Expr && lhs.head == :ref  # vars with []
					dsym = lhs.args[1]
					dsymref = "[$(lhs.args[2])]"
				else
					error("[backwardSweep] not a symbol on LHS of assigment $(e)") 
				end
				
				rhs = e.args[2]
				if typeof(rhs) == Expr
					println
					for i in 2:length(rhs.args)
						vsym = rhs.args[i]
						if contains(locals, vsym)
							push(lb, derive(rhs, i-1, :($dsym$dsymref)))
						end
					end
				elseif typeof(rhs) == Symbol
					dsym2 = symbol("__d$dsym$dsymref")
					vsym2 = symbol("__d$rhs")
					push(lb, :( $vsym2 = $dsym2) )
				else 
				end

			elseif e.head == :for  # TODO parse loops
				error("[backwardSweep] can't handle $(e.head) expressions") # TODO

			elseif e.head == :if  # TODO parse if structures
				error("[backwardSweep] can't handle $(e.head) expressions")  # TODO

			elseif e.head == :block
				e2 = backwardSweep(e, locals)
				push(lb, e2)
			else
				error("[backwardSweep] can't handle $(e.head) expressions")
			end
		end

		Expr(:block, reverse(lb), Any)
	end

	function derive(opex::Expr, index::Integer, dsym::Expr)
		op = opex.args[1]  # operator
		vsym = opex.args[1+index]
		vsym2 = symbol("__d$vsym")
		dsym2 = symbol("__d$dsym")

		if op == :+ 
			return :($vsym2 += $dsym2)

		elseif op == :- 
			if length(opex.args) == 2  # unary minus
				return :($vsym2 += -$dsym2)
			elseif index == 1
				return :($vsym2 += $dsym2)
			else
				return :($vsym2 += -$dsym2)
			end

		elseif op == :^
			if index == 1
				e = opex.args[3]
				if e == 2.0
					return :($vsym2 += 2 * $vsym * $dsym2)
				else
					return :($vsym2 += $e * $vsym ^ $(e-1) * $dsym2)
				end
			else
				v = opex.args[2]
				return :($vsym2 += log($v) * $v ^ $vsym * $dsym2)
			end

		elseif op == :*
			e = :(1.0)
			if index == 1
				return :($vsym2 += $dsym2 * transpose($(opex.args[3])))
			else
				return :($vsym2 += transpose($(opex.args[2])) * $dsym2)
			end

		elseif op == :dot
			e = index == 1 ? opex.args[3] : opex.args[2]
			return :($vsym2 += sum($e) .* $dsym2)

		elseif op == :log
			return :($vsym2 += $dsym2 ./ $vsym)

		elseif op == :exp
			return :($vsym2 += exp($vsym) .* $dsym2)

		elseif op == :/
			if index == 1
				e = opex.args[3]
				return :($vsym2 += $vsym ./ $e .* $dsym2)
			else
				v = opex.args[2]
				return :($vsym2 += - $v ./ ($vsym .* $vsym) .* $dsym2)
			end
		else
			error("[derive] Doesn't know how to derive operator $op")
		end
	end

##########  unfold Expr ##############
	function unfoldExpr(ex::Expr)
		assert(typeof(ex) == Expr, "[unfoldExpr] not an Expr $(ex)")
		
		if ex.head == :ref  # var[], send back directly
			return {ex}		
		elseif ex.head == :call
			lb = {}
			na = {ex.args[1]}   # function name
			ex2 = ex.args[2:end]

			# if more than 2 arguments, convert to nested expressions (easier for derivation)
			# TODO : probably not valid for all ternary, quaternary, etc.. operators, should work for *, +, sum
			while length(ex2) > 2
				a2 = pop(ex2)
				a1 = pop(ex2)
				push(ex2, Expr(:call, {ex.args[1], a1, a2}, Any))
			end

			for e in ex2  # e = :b
				if typeof(e) == Expr
					ue = unfoldExpr(e)
					for a in ue[1:end-1]
						push(lb, a)
					end
					nv = consume(nameTask)
					push(lb, :($nv = $(ue[end])))
					push(na, nv)
				else
					push(na, e)
				end
			end
			push(lb, Expr(ex.head, na, Any))
			return lb
		else
			error("[unfoldExpr] cannot handle $(ex)")
		end
	end

##########  var name factory   ##############
	function nameFactory()
		for i in 1:10000
			produce(symbol("__t$i"))
		end
	end

##########  helper function  to analyse expressions   #############

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
			assert(numel(init) == nparams, "$nparams initial values expected, got $(numel(init))")
			beta = init
		elseif typeof(init) <: Real
			beta = [ convert(Float64, init)::Float64 for i in 1:nparams]
		else
			error("cannot assign initial values (should be a Real or vector of Reals)")
		end

		#  log-likelihood function creation
		eval(quote
			function __loglik(beta::Vector{Float64})
				local acc
				$(Expr(:block, param_map, Any))
				acc = 0.0
				$model2
				return(acc)
			end
		end)

		#  first calc
		__lp = Main.__loglik(beta)
		__lp == -Inf ? error("Initial values out of model support, try other values") : nothing

		#  main loop
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

			#  Adaptive scaling using R.A.M. method
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

	function parseExpr(ex::Expr, assigns::Array{Any}, index::Integer)
		println(ex.head)

		# ex = model ; assigns = {} ; index = 1
		block = {}
		for e in ex.args  # e = ex.args[4]
			if !isa(e, Expr)
				push(block, e)

			elseif e.head == :(::)  #  model param declaration
				typeof(e.args[1]) == Symbol ? nothing : error("not a symbol on LHS of ~ : $(e.args[1])")
				par = e.args[1]  # param symbol defined here

				if e.args[2] == :scalar  #  simple decl : var::scalar
					push(assigns, :($(par) = __beta[$(index+1)]))
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
				e = :(__acc = _acc + sum(logpdf($(e.args[3]), $(e.args[2]))))
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

	nameTask = Task(nameFactory)

end  # end of module