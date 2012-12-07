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
				println("++++", e)
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
		println("---", block, "---")
		return(Expr(ex.head, block, Any), assigns, index)
	end

	parseExpr(ex::Expr) = parseExpr(ex, {}, 0) # initial function call

