module SimpleMCMC

export processExpr, expexp

const ACC_NAME = "__acc"
const PARAM_NAME = "__beta"
const TEMP_NAME = "__tmp"
const DERIV_PREFIX = "__d"

			  
##########  dispatcher main entry point  ############
const smap = [:(=) => :equal,
			  :(::) => :dcolon]

function processExpr(ex::Expr, action::Symbol, others...)
	if has(smap, ex.head) # stringify some symbols
		func = symbol("$(action)_$(smap[ex.head])")
	else
		func = symbol("$(action)_$(ex.head)")
	end

	mycall = expr(:call, func, expr(:quote, ex), others...)
	println("calling $mycall")
	eval(mycall)
#	if eval(:(method_exists($func, (Expr, map(typeof, others)...))))
#		eval(mycall)
#	else
#		error("[$action] can't process [$(ex.head)] expressions")
#	end
end

######## unfolding functions ###################
unfold(ex::Expr) = processExpr(ex, :unfold) # entry function

unfold_line(ex::Expr) = ex
unfold_ref(ex::Expr) = ex

unfold_block(ex::Expr) = expr(:block, map(x->processExpr(x, :unfold), ex.args))

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

######### active vars scanning functions  #############

function listVars(ex::Expr, avars) # entry function
	avars = Set{Symbol}(avars...)

	processExpr(ex, :listVars)
	avars
end

#function listVars_equal(ex::Expr, avars::Set{Symbol})
function listVars_equal(ex::Expr)
	ls = ex.args[2].args[2:end]
	ls = Set{Symbol}( filter(x -> isa(x, Symbol), ls)... )
	if length(intersect(ls, avars)) > 0 
		 return add(avars, ex.args[1])
	end
	avars
end

#listVars_line(ex::Expr, avars::Set{Symbol}) = avars
#listVars_ref(ex::Expr, avars::Set{Symbol}) = avars
listVars_line(ex::Expr) = nothing
listVars_ref(ex::Expr) = nothing
listVars_dcolon(ex::Expr) = nothing

#function listVars_block(ex::Expr, avars::Set{Symbol})
listVars_block(ex::Expr) = 
	map(x->processExpr(x, :listVars), ex.args)


######### model params scanning functions  #############

type Parmap
	map::Dict{Symbol, Expr}
	index::Integer
end

function findParams_generic(ex::Expr, params::Parmap)
	ls = filter(x -> isa(x, Expr), ex.args)
	for ex2 in ls
		params = processExpr(ex2, :findParams, params)
	end
	params
end

findParams_line(ex::Expr, params::Parmap) = params
findParams_ref(ex::Expr, params::Parmap) = params
findParams_call(ex::Expr, params::Parmap) = params

findParams_error(ex::Expr) = error("[findParams] can't process [$(ex.head)] expressions")

findParams_for(ex::Expr, params::Parmap) = findParams_error(ex)
findParams_if(ex::Expr, params::Parmap) = findParams_error(ex)
findParams_while(ex::Expr, params::Parmap) = findParams_error(ex)

findParams_block(ex::Expr, params::Parmap) = findParams_generic(ex, params)
findParams_equal(ex::Expr, params::Parmap) = findParams_generic(ex, params)

function findParams_dcolon(ex::Expr, params::Parmap)
	assert(typeof(ex.args[1]) == Symbol, 
		"not a symbol on LHS of :: $(ex.args[1])")
	par = ex.args[1]  # param symbol defined here
	def = ex.args[2]

	if def == :scalar  #  simple decl : var::scalar
		params.map[par] = :($PARAM_NAME[$(params.index+1)])
		params.index += 1

	elseif isa(def, Expr) && def.head == :call
		e2 = def.args
		if e2[1] == :vector
			nb = eval(e2[2])
			assert(isa(nb, Integer) && nb > 0, 
				"invalid size $(e2[2]) = $(nb)")

			params.map[par] = :($PARAM_NAME[$(params.index+1):$(nb+params.index)])
			params.index += nb
		else
			error("unknown parameter type $(e2[1])")
		end
	else
		error("unknown parameter expression $(ex)")
	end
	params
end


#function findParams(ex::Expr)
#	map::Dict{Symbol, Expr}
#	index::Integer
#
#	ex = processExpr(_findParams(ex)
#	if contains([:block, :(=)], ex.head
#
#end







##########  helper function  to analyse expressions   #############

	function expexp(ex::Expr, ident...)
		ident = (length(ident)==0 ? 0 : ident[1])::Integer
		println(rpad("", ident, " "), ex.head, " -> ")
		for e in ex.args
			typeof(e)==Expr ? expexp(e, ident+3) : println(rpad("", ident+3, " "), "[", typeof(e), "] : ", e)
		end
	end




end