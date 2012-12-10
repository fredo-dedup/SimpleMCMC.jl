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
				(ue, nv) = unfoldExpr(rhs)
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
				error("[unfoldBlock] can't parse $e")
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


##########  unfold Expr ##############
function unfoldExpr(ex::Expr)
	assert(typeof(ex) == Expr, "[unfoldExpr] not an Expr $(ex)")
	assert(ex.head == :call, "[unfoldExpr] call expected  $(ex)")

	lb = {}
	na = {ex.args[1]}   # function name
	for e in ex.args[2:end]  # e = :b
		if typeof(e) == Expr
			(ue, nv) = unfoldExpr(e)
			for a in ue[1:end-1]
				push(lb, a)
			end
			push(lb, :($nv = $(ue[end])))
			push(na, nv)
		else
			push(na, e)
		end
	end
	push(lb, Expr(ex.head, na, Any))
	(lb, consume(nameTask))
end

function nameFactory()
	for i in 1:10000
		produce(symbol("__t$i"))
	end
end

@assert unfoldExpr(:(4.0)) == ([], :(4.0)
@assert unfoldExpr(:(3 + x)) == :(:(), :(3+x))
@assert unfoldExpr(:(3*b + x)) == (:(t = 3*b), :(t+x))
@assert unfoldExpr(:(3 + x)) == :(4.0


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




