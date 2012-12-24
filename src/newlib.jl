module SimpleMCMC

export processExpr, expexp

const ACC_NAME = "__acc"
const PARAM_NAME = "__beta"
const TEMP_NAME = "__tmp"
const DERIV_PREFIX = "__d"

##########  main entry point  ############
function processExpr(ex::Expr, action::Symbol, others...)
	if ex.head == :(=) # stringify some symbols
		fname = "$(action)_equal"
	else
		fname = "$(action)_$(ex.head)"
	end
	mycall = expr(:call, symbol(fname), expr(:quote, ex), others...)
	println("calling $mycall")
	eval(mycall)
end

######## unfolding functions ###################
unfold_line(ex::Expr) = ex
unfold_ref(ex::Expr) = ex

unfold_error(ex::Expr) = error("[unfold] can't process [$(ex.head)] expressions")

unfold_for(ex::Expr) = unfold_error(ex)
unfold_if(ex::Expr) = unfold_error(ex)
unfold_while(ex::Expr) = unfold_error(ex)

unfold_block(ex::Expr) = map(x->processExpr(x, :unfold), ex.args)

function unfold_equal(ex::Expr)
	lhs = ex.args[1]
	assert(typeof(lhs) == Symbol ||  (typeof(lhs) == Expr && lhs.head == :ref),
		"[unfold] not a symbol on LHS of assigment $(ex)")

	rhs = ex.args[2]
	if typeof(rhs) == Symbol
		return expr(:(=), lhs, rhs)
	elseif typeof(rhs) == Expr
		ue = processExpr(rhs, :unfold)
		if isa(ue, Expr)
			return expr(:(=), lhs, rhs)
		elseif isa(ue, Tuple)
			lb = push(ue[1], :($lhs = $(ue[2])))
			return expr(:block, lb)
		end
	else  # unmanaged kind of lhs
	 	error("[unfold] can't handle RHS of assignment $ex")
	end
end

function unfold_call(ex::Expr)
	na = {ex.args[1]}   # function name
	args = ex.args[2:end]  # arguments

	# if more than 2 arguments, convert to nested expressions (easier for derivation)
	# TODO : probably not valid for all ternary, quaternary, etc.. operators, should work for *, +, sum
	println(args)
	while length(args) > 2
		a2 = pop(args)
		a1 = pop(args)
		push(args, expr(:call, ex.args[1], a1, a2))
	end

	lb = {}
	for e2 in args  # e2 = args[2]
		if typeof(e2) == Expr
			ue = processExpr(e2, :unfold)
			if isa(ue, Tuple)
				append!(lb, ue[1])
				lp = ue[2]
			else
				lp = ue
			end
			nv = gensym(TEMP_NAME)
			push(lb, :($nv = $(lp)))
			push(na, nv)
		else
			push(na, e2)
		end
	end

	return numel(lb)==0 ? expr(ex.head, na) : (lb, expr(ex.head, na))
end

######### interpretation pass functions  #############

type varNode
	pred::Vector{Union(varNode, Nothing)}
	succ::Union(varNode, Nothing}
end

varTree
function findActiveVars(ex::Expr, mvars::Array)
	assert(ex.head == :block, "[findActiveVars] not a block")

	avars = mvars  # will store active vars, i.e. dependent on model params & impacting __acc
	for e in ex.args  #  e  = ex.args[4]
		if e.head == :line   # line number marker, no treatment
		elseif e.head == :(=)  # assigment
			lhs = e.args[1]
			rhs = e.args[2]

			if typeof(rhs) == Symbol 
				rvars = {rhs}
			elseif (typeof(rhs) == Expr && rhs.head == :ref)
				rvars = {rhs.args[1]}
			elseif (typeof(rhs) == Expr && rhs.head == :call)
				rvars = rhs.args
			else  # unmanaged kind of lhs
			 	error("[findActiveVars] can't handle RHS of assignment $e")
			end

			if length(intersect(Set(rvars...), Set(avars...))) > 0
				if typeof(lhs) == Symbol # simple var case
					push(avars, lhs)
				elseif typeof(lhs) == Expr && lhs.head == :ref  # vars with []
					push(avars, lhs.args[1])
				else
					error("[findActiveVars] not a symbol on LHS of assigment $(e)") 
				end
			end

		elseif e.head == :for  # TODO parse loops
			error("[findActiveVars] can't handle $(e.head) expressions")  # TODO
		elseif e.head == :if  # TODO parse if structures
			error("[findActiveVars] can't handle $(e.head) expressions")  # TODO
		elseif e.head == :block # never used ?
			avars = unfoldBlock(e, avars)
		else
			error("[findActiveVars] can't handle $(e.head) expressions")
		end
	end

	avars
end












##########  helper function  to analyse expressions   #############

	function expexp(ex::Expr, ident...)
		ident = (length(ident)==0 ? 0 : ident[1])::Integer
		println(rpad("", ident, " "), ex.head, " -> ")
		for e in ex.args
			typeof(e)==Expr ? expexp(e, ident+3) : println(rpad("", ident+3, " "), "[", typeof(e), "] : ", e)
		end
	end




end