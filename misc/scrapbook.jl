 load("pkg")

 Pkg.init()

Pkg.add("Distributions")
Pkg.update("Distributions")


whos()


model = quote
	x::scalar

	resid = (x-2)*(x-1)
	resid ~ Normal(0, 0.01)  
end

myf, np = SimpleMCMC.buildFunctionWithGradient(model)
np
myf

eval(myf)

l0, grad = __loglik([1.])
l0, grad = __loglik([1.01])
l0, grad = __loglik([2.])
l0, grad = __loglik([2.01])
l0, grad = __loglik([1.99])
l0, grad = __loglik([1.01])

################################
Y = [1., 2, 3, 4]
X = [0. 1; 0 1; 1 1; 1 2]

model = quote
	vars::real(2)

	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  
end

include("../src/SimpleMCMC.jl")
myf, np = SimpleMCMC.buildFunctionWithGradient(model)
eval(myf)

res = SimpleMCMC.simpleHMC(model,100, 5, 1.5, 10, 0.1)

l0, grad = __loglik(ones(2))

[ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]


vars = ones(2)
resid=Y - X * vars
dresid = zeros(4)
dresid += /(sum(+(-(resid),0)),1.0)
dvars = zeros(2) 
dvars += transpose(X) * dresid


################################

myf, np = SimpleMCMC.buildFunctionWithGradient(model)
eval(myf)

l0, grad = __loglik(ones(nbeta))
[ [ (beta=ones(nbeta) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:nbeta ] grad]



################################

model = quote
	x::real
	x ~ Normal(0., 1.)  
end

include("../src/SimpleMCMC.jl")
load("../../Distributions.jl/src/Distributions")
using Distributions
import Distributions.logpdf

push
myf, np = SimpleMCMC.buildFunction(model)
eval(myf)
myf, np = SimpleMCMC.buildFunctionWithGradient(model)
eval(myf)
SimpleMCMC.expexp(myf)
SimpleMCMC.expexp(:( Distributions.normal(0,1)))

eval(myf.args[2].args[3])
__beta = ones(10)
vars = __beta[1:10]
__acc = 1.0
__acc = +(__acc, sum(Distributions.logpdf(Distributions.Normal(0, 1.0), vars)))
resid = -(Y, *(X, vars))
__acc = +(__acc, sum(Distributions.logpdf(Distributions.Normal(0, 1.0), resid)))
return __acc


__loglik(ones(10))
test = :__loglik
Main.eval(:( $test(ones(10))))
SimpleMCMC.expexp(:((Main.($test))(ones(10))))


__beta = ones(10)
__loglik([1.01])


res = SimpleMCMC.simpleRWM(model,10000)
res = SimpleMCMC.simpleHMC(model, 10000, 2, 0.1)
res = SimpleMCMC.simpleHMC(model, 10000, 4, 5e-3)
writedlm("/tmp/dump.txt", res)

################################
model = quote
	b::real
	k::real(5)
	
	a = b+6
	x = sin(dot(k, z))

	x ~ Weibull(a, 2.0)
end

SimpleMCMC.buildFunctionWithGradient(model)
include("../src/SimpleMCMC.jl")

myf, np, pmap = SimpleMCMC.parseModel(model, true)
dump(myf)
SimpleMCMC.unfold(myf)
ex=myf



	(model2, nparams, pmap) = SimpleMCMC.parseModel(model, true)
	exparray = SimpleMCMC.unfold(model2)
	avars = SimpleMCMC.listVars(exparray, keys(pmap))
	dmodel = SimpleMCMC.backwardSweep(exparray, avars)

	# build body of function
	body = { expr(:(=), k, v) for (k,v) in pairs(pmap)}

	push!(body, :($(SimpleMCMC.ACC_SYM) = 0.)) 

	body = vcat(body, exparray)

	push!(body, :($(symbol("$(SimpleMCMC.DERIV_PREFIX)$(SimpleMCMC.ACC_SYM)")) = 1.0))
	delete!(avars, SimpleMCMC.ACC_SYM) # remove accumulator, treated above
	for v in avars
		push!(body, :($(symbol("$DERIV_PREFIX$v")) = zero($(symbol("$v")))))
	end


############################################################################

model = quote
    b::real(3)
    
    b[2] ~ Normal(0,1)
    z = b * X
    z = 4
    y = z
    z ~  Uniform(4,z,y)
end

include("../src/SimpleMCMC.jl")

(model2, nparams, pmap) = SimpleMCMC.parseModel(model, true)
exparray = SimpleMCMC.unfold(model2)
    avars = listVars(exparray, keys(pmap))
    dmodel = backwardSweep(exparray, avars)

    subst = Dict{Symbol, Symbol}()
    used = Array(Symbol,0)
    el = copy(exparray)
    for idx in 1:length(el) # idx, el = 8, exparray
    	ex2 = el[idx]
        lhs = elements(SimpleMCMC.getSymbols(ex2.args[1]))[1]  # there should be only one
        rhs = SimpleMCMC.getSymbols(ex2.args[2])

        if isa(etype(el[idx]), Exprequal)
        	if isa(el[idx].args[2], Symbol) # simple assign
        		if has(subst, el[idx].args[2])
        			el[idx].args[2] = subst[el[idx].args[2]]
        		end
        	elseif isa(el[idx].args[2], Real) # if number do nothing
        	elseif isa(etype(el[idx].args[2]), Exprref) # simple assign of a ref
        		if has(subst, el[idx].args[2].args[1])
        			el[idx].args[2].args[1] = subst[el[idx].args[2].args[1]]
        		end
        	elseif isa(etype(el[idx].args[2]), Exprcall) # function call
        		for i in 2:length(el[idx].args[2].args) # i=4
	        		if has(subst, el[idx].args[2].args[i])
	        			el[idx].args[2].args[i] = subst[el[idx].args[2].args[i]]
	        		end
	        	end
	        else
	        	error("[unfold] can't subsitute var name in $ex2")
	        end
	    else
	    	error("[unfold] not an assignment ! : $ex2")
	    end

        if contains(used, lhs) # var already set once
            subst[lhs] = gensym("$lhs")
        	if isa(el[idx].args[1], Symbol) # simple assign
        		if has(subst, el[idx].args[1])
        			el[idx].args[1] = subst[el[idx].args[1]]
        		end
        	elseif isa(etype(el[idx].args[1]), Exprref) # simple assign of a ref
        		if has(subst, el[idx].args[1].args[1])
        			el[idx].args[1].args[1] = subst[el[idx].args[1].args[1]]
        		end
	        else
	        	error("[unfold] can't subsitute var name in $lhs")
	        end
        else # var set for the first time
            push!(used, lhs)
        end

    end

exparray 
el
