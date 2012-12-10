############# big lm  10.000 obs x 40 var  ###########
load("SimpleMCMC.jl")
import SimpleMCMC.simpleRWM, SimpleMCMC.parseExpr, SimpleMCMC.expexp

load("distributions.jl") 
import Distributions.Gamma, Distributions.Normal, Distributions.Exponential
import Distributions.Poisson 
import Distributions.pdf, Distributions.logpdf

begin
	srand(1)
	n = 10000
	nbeta = 40
	X = [fill(1, (n,)) randn((n, nbeta-1))]
	beta0 = randn((nbeta,))
	Y = X * beta0 + randn((n,))
end

model = quote
	vars::vector(nbeta)

	vars ~ Normal(0, 1)
	resid = Y - X * vars
	resid ~ Normal(0, 1)
end

res = simpleRWM(model, 1000)

dlmwrite("c:/temp/mjcl.txt", res)


(model2, assgmt, nparams) = parseExpr(model)

expexp(model2)
expexp(:(a+b+c))
expexp(:(sum(a,b,c)))
expexp(:(if a==b ; c=d; elseif a<b ; c=2d; else c=3d; end))
expexp(:(for i in 1:10; b=3.0;end))
expexp(:(while d > 4; d+= 1;end))

ex = model2
args2 = Vector{Any}

type pBlock
	block::Vector{Expr}
	lvars::Vector{Symbol}
	evars::Vector{Symbol}
end
pBlock() = pBlock([Expr(:line, {0} , Any)], [:none], [:none])
# pBlock() = pBlock([Expr(:line, {0} , Any)], {}, {})

a = pBlock()
a

##########  unfoldBlock ##############

function unfoldBlock(ex::Expr)
	assert(ex.head == :block, "[unfoldBlock] not a block")

	lb = {}
	for e in ex.args  #  e  = ex.args[2]
		if e.head == :line   # line number marker, no treatment
			push(lb, e)
		elseif e.head == :(=)  # assigment
			assert(typeof(e.args[1]) == Symbol, "not a symbol on LHS of assigment $(e)") # TODO : manage symbol[]
			
			rhs = e.args[2]
			if typeof(rhs) == Expr
				ue = unfoldExpr(rhs)
				if length(ue)==1 # no variable creation necessary
					push(lb, e)
				else
					for a in ue[1:end-1]
						push(lb, a)
					end
					push(lb, :($(e.args[1]) = $(ue[end])))
				end
			elseif typeof(rhs) == Symbol
				push(lb, e)
			else
				push(lb, e)
			# 	error("[unfoldBlock] can't parse $e")
			end

		elseif e.head == :for  # TODO parse loops

		elseif e.head == :while  # TODO parse loops

		elseif e.head == :if  # TODO parse if structures

		elseif e.head == :block
			e2 = unfoldBlock(e)
			push(lb, e2)
		else
			error("[unfoldBlock] can't handle $(e.head) expressions")
		end
	end

	Expr(:block, lb, Any)
end

function localVars(ex::Expr)
	assert(ex.head == :block, "[localVars] not a block")

	lb = {}
	for e in ex.args  #  e  = ex.args[2]
		if e.head == :(=)  # assigment
			assert(typeof(e.args[1]) == Symbol, "[localVars] not a symbol on LHS of assigment $(e)") # TODO : manage symbol[]
			
			push(lb, e.args[1])

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
			dsym = e.args[1]
			assert(typeof(dsym) == Symbol, "[backwardSweep] not a symbol on LHS of assigment $(e)") # TODO : manage symbol[]
			
			rhs = e.args[2]
			if typeof(rhs) == Expr
				for i in 2:length(rhs.args)
					vsym = rhs.args[i]
					if contains(locals, vsym)
						push(lb, derive(rhs, i-1, dsym))
					end
				end
			elseif typeof(rhs) == Symbol
				dsym2 = symbol("__d$dsym")
				vsym2 = symbol("__d$rhs")
				push(lb, :( $vsym2 = $dsym2) )
			else 
			end

		elseif e.head == :for  # TODO parse loops
			# e2 = backwardSweep(e)
			# push(lb, e2)

		elseif e.head == :while  # TODO parse loops
			# e2 = backwardSweep(e)
			# push(lb, e2)

		elseif e.head == :if  # TODO parse if structures
			# e2 = backwardSweep(e.args[2])
			# push(lb, e2)

		elseif e.head == :block
			e2 = backwardSweep(e, locals)
			push(lb, e2)
		else
			error("[backwardSweep] can't handle $(e.head) expressions")
		end
	end

	Expr(:block, reverse(lb), Any)
end

function derive(opex::Expr, index::Integer, dsym::Symbol)
	op = opex.args[1]  # operator
	vsym = opex.args[1+index]
	vsym2 = symbol("__d$vsym")
	dsym2 = symbol("__d$dsym")

	if op == :+
		return :($vsym2 += $dsym2)
	elseif op == :-
		return :($vsym2 += $(index==1 ? "+" : "-") $dsym2)
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
		for i in 2:length(opex)
			if i != index+1
				if i < index+1
					e = :($e * $(opex.args[i])')
				else	
					e = :($e * $(opex.args[i]))
				end
			end
		return :($vsym2 += $e * $dsym2)
	elseif op == :log
		return :($vsym2 += $dsym2 / $vsym)
	elseif op == :/
		if index == 1
			e = opex.args[3]
			return :($vsym2 += $vsym / $e * $dsym2)
		else
			v = opex.args[2]
			return :($vsym2 += - $v / ($vsym*$vsym) * $dsym2)
		end
	else
		error("Can't derive operator $op")
	end
end

##########  unfold Expr ##############
function unfoldExpr(ex::Expr)
	assert(typeof(ex) == Expr, "[unfoldExpr] not an Expr $(ex)")
	assert(ex.head == :call, "[unfoldExpr] call expected  $(ex)")

	lb = {}
	na = {ex.args[1]}   # function name
	for e in ex.args[2:end]  # e = :b
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
	lb
end

function nameFactory()
	for i in 1:10000
		produce(symbol("__t$i"))
	end
end

nameTask = Task(nameFactory)
@assert isequal(unfoldExpr(:(3 + x)), [:(3+x)])
@assert isequal(unfoldExpr(:(3*b + x)), [:(__t1 = 3*b), :(__t1 + x)])


##########  tests ##############

nameTask = Task(nameFactory)
# consume(nameTask)

ex=:(a+b)
unfoldExpr(:(a+b))
unfoldExpr(:(3*a+b))

ex = quote
	a = b +c
	d = 2f - sin(y)
	ll = log(a+d)
end

unfoldBlock(ex)

ex = quote
	a = b +c
	begin
		z = 3
		k = z*z + x
	end
	d = 2f - sin(y)
	ll = log(a+d)
end

unfoldBlock(ex)

ex = quote
	(y = x + 13 ; z = y^2 + x ; ll = z)
end

ex2 = unfoldBlock(ex)

vars = localVars(ex2)
backwardSweep(ex2, vars)

expexp(:(h'))
