module SimpleMCMC

export expexp

const ACC_NAME = :__acc
const PARAM_SYM = :__beta
const TEMP_NAME = "__tmp"
const DERIV_PREFIX = "__d"

			  
##########  dispatcher main entry point  ############
const emap = [:(=) => :Exprequal,
			  :(::) => :Exprdcolon]

for ex in [:equal, :dcolon, :call, :block, :for, :if, :ref, :line]
	nt = symbol(strcat("Expr", ex))
	eval(quote
		type $nt
			head::Symbol
			args::Vector
			typ::Any
		end
		($nt)(ex::Expr) = ($nt)(ex.head, ex.args, ex.typ)
		toExpr(ex::$nt) = expr(ex.head, ex.args)
	end)
end

function etype(ex::Expr)
	nt = has(emap, ex.head) ? emap[ex.head] : symbol(strcat("Expr", ex.head))
#	println("$(ex.head) - $ex")
	eval(:(($nt)($(expr(:quote, ex)))))
end

######## unfolding functions ###################

function unfold(ex::Expr)

	unfold(ex::Exprline) = nothing
	unfold(ex::Exprref) = toExpr(ex)
	function unfold(ex::Exprblock)), ex.args)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = unfold(etype(ex2))
				ex3==nothing ? nothing : push(al, ex3)
			else
				push(al, ex2)
			end
		end
		expr(ex.head, al)
	end

	function unfold(ex::Exprequal)
		lhs = ex.args[1]
		assert(typeof(lhs) == Symbol ||  (typeof(lhs) == Expr && lhs.head == :ref),
			"[unfold] not a symbol on LHS of assigment $(ex)")

		rhs = ex.args[2]
		if typeof(rhs) == Symbol
			return expr(:(=), lhs, rhs)
		elseif typeof(rhs) == Expr
			ue = unfold(rhs)
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

	function unfold(ex::Exprcall)
		na = {ex.args[1]}   # function name
		args = ex.args[2:end]  # arguments

		# if more than 2 arguments, convert to nested expressions (easier for derivation)
		# TODO : probably not valid for all ternary, quaternary, etc.. operators, should work for *, +, sum
		while length(args) > 2
			a2 = pop(args)
			a1 = pop(args)
			push(args, expr(:call, ex.args[1], a1, a2))
		end

		lb = {}
		for e2 in args  # e2 = args[2]
			if typeof(e2) == Expr
				ue = unfold(e2)
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

	unfold(etype(ex))
end


######### active vars scanning functions  #############

function listVars(ex::Expr, avars) # entry function
	avars = Set{Symbol}(avars...)

	function listVars(ex::Exprequal)
		ls = ex.args[2].args[2:end]
		lhs = 
		if isa(ex.args[2])
		for ex2 in 
		ls = Set{Symbol}( filter(x -> isa(x, Symbol), ls)... )
		if length(intersect(ls, avars)) > 0 
			 return add(avars, ex.args[1])
		end
	end

	listVars(ex::Exprline) = nothing
	listVars(ex::Exprref) = nothing
	listVars(ex::Exprdcolon) = nothing
	listVars(ex::Exprblock) = map(x->listVars(etype(x)), ex.args)

	listVars(etype(ex))
	avars
end

######### model params scanning functions  #############

function findParams(ex::Expr)

	function findParams_generic(ex)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = findParams(etype(ex2))
				ex3==nothing ? nothing : push(al, ex3)
			else
				push(al, ex2)
			end
		end
		expr(ex.head, al)
	end

	findParams(ex::Exprline) = toExpr(ex)
	findParams(ex::Exprref) = toExpr(ex)
	findParams(ex::Exprcall) = toExpr(ex)
	findParams(ex::Exprblock) = findParams_generic(ex)
	findParams(ex::Exprequal) = findParams_generic(ex)

	function findParams(ex::Exprdcolon)
		assert(typeof(ex.args[1]) == Symbol, 
			"not a symbol on LHS of :: $(ex.args[1])")
		par = ex.args[1]  # param symbol defined here
		def = ex.args[2]

		if def == :scalar  #  simple decl : var::scalar
			pmap[par] = :($PARAM_SYM[$(index+1)])
			index += 1

		elseif isa(def, Expr) && def.head == :call
			e2 = def.args
			if e2[1] == :vector
				nb = eval(e2[2])
				assert(isa(nb, Integer) && nb > 0, 
					"invalid vector size $(e2[2]) = $(nb)")

				pmap[par] = :($PARAM_SYM[$(index+1):$(nb+index)])
				index += nb
			else
				error("unknown parameter type $(e2[1])")
			end
		else
			error("unknown parameter expression $(ex)")
		end
		nothing
	end

	pmap = Dict{Symbol, Expr}()
	index = 0

	ex = findParams(etype(ex))
	(ex, index, pmap)
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