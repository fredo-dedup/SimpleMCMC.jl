
unfold_line(ex::Expr) = ex
unfold_for(ex::Expr) = error("[unfold] can't process $(ex.head) expressions")
unfold_if(ex::Expr) = error("[unfold] can't process $(ex.head) expressions")
unfold_while(ex::Expr) = error("[unfold] can't process $(ex.head) expressions")
unfold_block(ex::Expr) = error("[unfold] can't process $(ex.head) expressions")
unfold_ref(ex::Expr) = {ex}

function unfold_equal(ex::Expr)
	lb = {}

	lhs = ex.args[1]
	assert(typeof(lhs) == Symbol ||  (typeof(lhs) == Expr && lhs.head == :ref),
		"[unfold] not a symbol on LHS of assigment $(ex)")

	rhs = ex.args[2]
	if typeof(rhs) == Symbol ||  (typeof(rhs) == Expr && rhs.head == :ref)
		push(lb, ex)
	elseif typeof(rhs) == Expr
		ue = processExpr(rhs, :unfold)
		if length(ue)==1 # simple expression, no variable creation necessary
			push(lb, ex)
		else # several nested expressions
			for a in ue[1:end-1]
				push(lb, a)
				lhs2 = a.args[1]
				if typeof(lhs2) == Symbol # simple var case
				elseif typeof(lhs2) == Expr && lhs2.head == :ref  # vars with []
				else
					error("[unfold] not a symbol on LHS of assigment $(e)") 
				end
			end
			push(lb, :($lhs = $(ue[end])))
		end
	else  # unmanaged kind of lhs
	 	error("[unfold] can't handle RHS of assignment $e")
	end

	lb
end

function unfold_call(ex::Expr)
	lb = {}
	na = {ex.args[1]}   # function name
	args = ex.args[2:end]  # arguments

	# if more than 2 arguments, convert to nested expressions (easier for derivation)
	# TODO : probably not valid for all ternary, quaternary, etc.. operators, should work for *, +, sum
	while length(args) > 2
		a2 = pop(args)
		a1 = pop(args)
		push(args, Expr(:call, {ex.args[1], a1, a2}, Any))
	end

	for e2 in args  # e2 = args[2]
		if typeof(e2) == Expr
			ue = processExpr(e2, :unfold)
			for a in ue[1:end-1]
				push(lb, a)
			end
			nv = gensym("temp")
			push(lb, :($nv = $(ue[end])))
			push(na, nv)
		else
			push(na, e2)
		end
	end
	push(lb, Expr(ex.head, na, Any))
	lb
end

##########  main entry point  ############
function processExpr(ex::Expr, action::Symbol)
	if ex.head == :(=) # stringify some symbols
		fname = "$(action)_equal"
	else
		fname = "$(action)_$(ex.head)"
	end
	mycall = Expr(:call, {symbol(fname), expr(:quote, ex)}, Any)
	# mycall = :($(fname)($(expr(:quote, ex))))
	# apply(symbol(fname), ex)
	# mycall.args[2] = quote
	# 	$ex
	# end
	println("caling $mycall")
	eval(mycall)
end
