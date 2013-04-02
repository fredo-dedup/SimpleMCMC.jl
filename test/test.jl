#########################################################################
#    testing script for gradients
#########################################################################

include("../src/SimpleMCMC.jl")


## variables of different dimension for testing
v0ref = 2.
v1ref = [2., 3, 0.1, 0, -5]
v2ref = [-1. 3 0 ; 0 5 -2]

## error thresholds
DIFF_DELTA = 1e-9
ERROR_THRESHOLD = 2e-2

good_enough(x,y) = isfinite(x) ? (abs(x-y) / max(ERROR_THRESHOLD, abs(x))) < ERROR_THRESHOLD : isequal(x,y) 
good_enough(t::Tuple) = good_enough(t[1], t[2])

##  gradient check by comparing numerical gradient to automated gradient
function deriv1(ex::Expr, x0::Union(Float64, Vector{Float64}, Matrix{Float64})) #  ex= :(-x) ; x0 = 1.0
	println("testing gradient of $ex at x = $x0")

	nx = length(x0)  # nx=3
	dx = length(size(x0))
	if dx == 0
		pexpr = :( x::real )
	elseif dx == 1  # vector
		pexpr = :( x::real($nx) )
	elseif dx == 2  # matrix
		pexpr = :( x::real($(size(x0,1)), $(size(x0, 2))) )
	else
		error("x0 should have up to 2 dimensions")
	end

	model = expr(:block, pexpr, :(y = $ex), :(y ~ TestDiff()))

	myf, np = SimpleMCMC.buildFunctionWithGradient(model)

	__beta = vec([x0])
	l0, grad0 = myf(__beta)  

	gradn = zeros(nx)
	for i in 1:nx 
		__beta = vec([x0])
		__beta[i] += DIFF_DELTA
		l, grad = myf(__beta)  
		gradn[i] = (l-l0)/DIFF_DELTA
	end

	assert(all(good_enough, zip([grad0], [gradn])),
		"Gradient false for $ex at x=$x0, expected $(round(gradn,5)), got $(round(grad0,5))")
end

## argument pattern generation for testing
# all args can be scalar, vector or matrices, but with compatible dimensions (i.e same size for arrays)
function testpattern1(parnames, rules)
	arity = length(parnames)
	ps = [ ifloor((i-1) / 2^(j-1)) % 2 for i=1:2^arity, j=1:arity]
	vcat(ps, 2*ps[2:end,:])
end

# all args are scalar
function testpattern2(parnames, rules)
	arity = length(parnames)
	zeros(Int32, 1, arity)
end

# all args are vectors
function testpattern3(parnames, rules)
	arity = length(parnames)
	ones(Int32, 1, arity)
end

# all args are matrices
function testpattern4(parnames, rules)
	arity = length(parnames)
	ones(Int32, 1,arity)*2
end

# all args can be scalar, vector or matrices, but with no more than one array
function testpattern5(parnames, rules)
	arity = length(parnames)
	ps = [ ifloor((i-1) / 2^(j-1)) % 2 for i=1:2^arity, j=1:arity]
	ps = vcat(ps, 2*ps[2:end,:])
	ps[ps * ones(arity) .<= 1, :]
end

## runs arg dimensions combinations
function runpattern(fsym, parnames, rules, combin)
	arity = length(parnames)

	for ic in 1:size(combin,1)  # try each arg dim in combin
		c = combin[ic,:]
		par = [ symbol("arg$i") for i in 1:arity]

		# create variables
		for i in 1:arity  # generate arg1, arg2, .. variables
			vn = par[i]  
			vref = [:v0ref, :v1ref, :v2ref][c[i]+1]
			eval(:( $vn = copy($vref)))
		end

		# apply transformations on args
		for r in rules # r = rules[1]
			if isa(r, Expr) && r.head == :(->)
				pos = find(parnames .== r.args[1]) # find the arg the rules applies to 
				assert(length(pos)>0, "arg of rule ($r.args[1]) not found among $parnames")
				vn = symbol("arg$(pos[1])")
				eval(:( $vn = map($r, $vn)))
			end
		end

		# now run tests
		prange = any(rules .== :(:exceptlast)) ? (1:(arity-1)) : (1:arity)
		# println("$prange - $(length(rules)) - $rules")
		for p in prange  # try each argument as parameter
			tpar = copy(par)
			tpar[p] = :x  # replace tested args with parameter symbol :x for deriv1 testing func
			f = expr(:call, [fsym, tpar...]...) 
			vn = symbol("arg$p")
			x0 = eval(vn)  # set x0 for deriv 1
			deriv1(f, x0+0.001)  # shift sightly (to avoid numerical derivation pb on max() and min())
		end
	end
end

## macro to simplify tests expression
macro mtest(pattern, func::Expr, constraints...)
	tmp = [ isa(e, Symbol) ? expr(:quote, e) : e for e in constraints]
	quote
		local fsym = $(expr(:quote, func.args[1]))
		local pars = $(expr(:quote, [func.args[2:end]...]) ) 
		local rules = $(expr(:quote, [tmp...]) ) 

		combin = ($pattern)(pars, rules)
		runpattern(fsym, pars, rules, combin)
	end
end

#########################################################################
#  tests on functions
#########################################################################

## regular functions
@mtest testpattern1 x+y 
@mtest testpattern1 x+y+z 
@mtest testpattern1 sum(x)
@mtest testpattern1 x-y
@mtest testpattern1 x.*y
@mtest testpattern1 x./y  y -> y==0 ? 0.1 : y
@mtest testpattern1 x.^y  x -> x<=0 ? 0.2 : x 
@mtest testpattern1 sin(x)
@mtest testpattern1 abs(x)
@mtest testpattern1 cos(x)
@mtest testpattern1 exp(x)
@mtest testpattern1 log(x) x -> x<=0 ? 0.1 : x

@mtest testpattern1 transpose(x) 
deriv1(:(x'), [-3., 2, 0]) 

@mtest testpattern1 max(x,y) 
@mtest testpattern1 min(x,y)

@mtest testpattern2 x^y 

@mtest testpattern5 x/y   y->y==0 ? 0.1 : y

@mtest testpattern5 x*y 
tz = transpose(v1ref)
deriv1(:(x*tz), [-3., 2, 0]) 
deriv1(:(tz*x), v1ref)  
deriv1(:(v2ref*x), [-3., 2, 0])
deriv1(:(v2ref[:,1:2]*x), [-3. 2 0 ; 1 1 -2]) 

@mtest testpattern2 dot(x,y) 
@mtest testpattern3 dot(x,y) 

## distributions
@mtest testpattern1 SimpleMCMC.logpdfNormal(mu,sigma,x)  sigma -> sigma<=0 ? 0.1 : sigma
@mtest testpattern1 SimpleMCMC.logpdfWeibull(sh,sc,x)    sh->sh<=0?0.1:sh  sc->sc<=0?0.1:sc  x->x<=0?0.1:x
@mtest testpattern1 SimpleMCMC.logpdfUniform(a,b,x)      a->a-10 b->b+10
@mtest testpattern1 SimpleMCMC.logpdfBeta(a,b,x)         x->min(0.99, max(0.01, x)) a->a<=0?0.1:a b->b<=0?0.1:b

# note for Bernoulli : having prob=1 or 0 is ok but will make the numeric differentiator 
#  of deriv1 fail => not tested
@mtest testpattern1 SimpleMCMC.logpdfBernoulli(prob,x)   exceptlast prob->min(0.99, max(0.01, prob)) x->(x>0)+0. 

@mtest testpattern1 SimpleMCMC.logpdfPoisson(l,x)   exceptlast l->l<=0?0.1:l x->iround(abs(x)) 


# fails, should not test parameter on 'size'
@mtest testpattern1 SimpleMCMC.logpdfBinomial(size, prob,x)   exceptlast prob->min(0.99, max(0.01, prob)) x->(x>0)+0. 


# for x in 0.1:0.1:1.0
# 	println(SimpleMCMC.logpdfBeta(2,5, x))
# end
# SimpleMCMC.logpdfBeta(0.5, 0.5, 0.1)





#########################################################################
#   misc. tests
#########################################################################

# Parsing should throw an error on generating the gradient code 
try
	deriv1(:(SimpleMCMC.logpdfBernoulli(1, x)), [0.])
	throw("no error !!")
catch e
	assert(e != "no error !!", 
		"parser not throwing error when logpdfBernoulli has a parameter dependant sampled variable")
end

##  ref  testing
deriv1(:(x[2]),              v1ref)
deriv1(:(x[2:3]),            v1ref)
deriv1(:(x[2:end]),          v1ref)

deriv1(:(x[2:end]),          v2ref)
deriv1(:(x[2]),              v2ref)
deriv1(:(x[2:4]),            v2ref)
deriv1(:(x[:,2]),            v2ref)
deriv1(:(x[1,:]),            v2ref)
deriv1(:(x[2:end,:]),        v2ref)
deriv1(:(x[:,2:end]),        v2ref)

deriv1(:(x[2]+x[1]),          v2ref)
deriv1(:(log(x[2]^2+x[1]^2)), v2ref)

# fail case when individual elements of an array are set several times
# FIXME : correct var renaming step in unfold...
# model = :(x::real(3); y=x; y[2] = x[1] ; y ~ TestDiff())

