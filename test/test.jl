#########################################################################
#    testing script for gradients
#########################################################################

include("../src/SimpleMCMC.jl")

DIFF_DELTA = 1e-9
ERROR_THRESHOLD = 1e-2

good_enough(x,y) = isfinite(x) ? (abs(x-y) / max(ERROR_THRESHOLD, abs(x))) < ERROR_THRESHOLD : isequal(x,y) 
good_enough(t::Tuple) = good_enough(t[1], t[2])

############# gradient checking function  ######################
# compares numerical gradient to automated gradient

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
	# ex2 = myf.args[2] # use body of function only

	__beta = vec([x0])
	# l0, grad0 = eval(ex2)  
	l0, grad0 = myf(__beta)  

	gradn = zeros(nx)
	for i in 1:nx 
		__beta = vec([x0])
		__beta[i] += DIFF_DELTA
		# l, grad = eval(ex2) 
		l, grad = myf(__beta)  
		gradn[i] = (l-l0)/DIFF_DELTA
	end

	# println("------- expected $(round(gradn,5)), got $(round(grad0,5))")
	assert(all(good_enough, zip([grad0], [gradn])),
		"Gradient false for $ex at x=$x0, expected $(round(gradn,5)), got $(round(grad0,5))")
end

macro mult(func, myex, values)
	quote
		for xt in $values
			($func)($(expr(:quote, myex)), xt) 
		end
	end
end

## variables of different dimension for testing
v0ref = 2.
v1ref = [2., 3, 0.1, 0, -5]
v2ref = [-1. 3 0 ; 0 5 -2]

## testing functions, with constraints to avoid meaningless tests  (i.e. 1/x with x = 0)

###### argument pattern generation for testing ##########
## generates all possible combinations of argument dimensions
## with of one or several of them being the parameter for the gradient

# all args can be scalar, vector or matrices, but with compatible dimensions (i.e same size for arrays)
function testpattern1(parnames, rules)
	arity = length(parnames)
	ps = Int32[ ifloor((i-1) / 2^(j-1)) % 2 for i=1:2^arity, j=1:arity]
	ps = vcat(ps, 2*ps[2:end,:])
	pp = Int32[ ifloor((i-1) / 2^(j-1)) % 2 for i=2:2^arity, j=1:arity]
	combin = Any[]
	for i in 1:size(ps,1) # i = 5
		for j in 1:size(pp,1) # j = 6
			pos = find(pp[j,:] .== 1)
			ptyp = unique(ps[i, find(pp[j,:] .== 1)])
			if length(ptyp) == 1 # all params should have same # dims
				push!(combin, (ps[i,:], pp[j,:], ptyp[1]))
			end
		end
	end

	combin
end

# all args are scalar
function testpattern2(parnames, rules)
	arity = length(parnames)
	ps = zeros(Int32, 1,arity)
	pp = Int32[ ifloor((i-1) / 2^(j-1)) % 2 for i=2:2^arity, j=1:arity]
	combin = Any[]
	for i in 1:size(ps,1) # i = 5
		for j in 1:size(pp,1) # j = 6
			ptyp = unique(ps[i, find(pp[j,:] .== 1)])
			push!(combin, (ps[i,:], pp[j,:], ptyp[1]))
		end
	end

	combin
end

# all args are vectors
function testpattern3(parnames, rules)
	arity = length(parnames)
	ps = ones(Int32, 1,arity)
	pp = Int32[ ifloor((i-1) / 2^(j-1)) % 2 for i=2:2^arity, j=1:arity]
	combin = Any[]
	for i in 1:size(ps,1) # i = 5
		for j in 1:size(pp,1) # j = 6
			ptyp = unique(ps[i, find(pp[j,:] .== 1)])
			push!(combin, (ps[i,:], pp[j,:], ptyp[1]))
		end
	end

	combin
end

# all args are matrices
function testpattern4(parnames, rules)
	arity = length(parnames)
	ps = ones(Int32, 1,arity)*2
	pp = Int32[ ifloor((i-1) / 2^(j-1)) % 2 for i=2:2^arity, j=1:arity]
	combin = Any[]
	for i in 1:size(ps,1) # i = 5
		for j in 1:size(pp,1) # j = 6
			ptyp = unique(ps[i, find(pp[j,:] .== 1)])
			push!(combin, (ps[i,:], pp[j,:], ptyp[1]))
		end
	end

	combin
end

# all args can be scalar, vector or matrices, but with no more than one array
function testpattern5(parnames, rules)
	arity = length(parnames)
	ps = Int32[ ifloor((i-1) / 2^(j-1)) % 2 for i=1:2^arity, j=1:arity]
	ps = ps[ps * ones(arity) .<= 1, :]
	ps = vcat(ps, 2*ps[2:end,:])
	pp = Int32[ ifloor((i-1) / 2^(j-1)) % 2 for i=2:2^arity, j=1:arity]
	combin = Any[]
	for i in 1:size(ps,1) # i = 5
		for j in 1:size(pp,1) # j = 6
			pos = find(pp[j,:] .== 1)
			ptyp = unique(ps[i, find(pp[j,:] .== 1)])
			if length(ptyp) == 1 # all params should have same # dims
				push!(combin, (ps[i,:], pp[j,:], ptyp[1]))
			end
		end
	end

	combin
end


function runpattern(fsym::Union(Symbol,Expr), parnames, rules, combin)
	# fsym = :exp ; parnames = [:x] ; rules=[:(y->y==0?0.1:y)] 
	#  fsym= :(SimpleMCMC.logpdfBernoulli); parnames= [:prob,:x] ;rules=[:( prob->min(0.99, max(0.01, prob))), :(x->(x>0)+0.)]
	arity = length(parnames)

	for p in combin  # p = combin[1]
		# par = [[:v0, :v1, :v2][p[1][i]+1] for i in 1:arity]
		par = [ symbol("arg$i") for i in 1:arity]

		# create variables
		for i in 1:arity  # generate arg1, arg2, .. variables
			vn = par[i]  
			vref = [:v0ref, :v1ref, :v2ref][p[1][i]+1]
			eval(:( $vn = copy($vref)))
		end

		# apply transformations on args
		for c in rules # c = rules[1]
			if isa(c, Expr) && c.head == :(->)
				pos = find(parnames .== c.args[1]) # find the arg the rules applies to 
				assert(length(pos)>0, "arg of rule not found among $parnames")
				vn = symbol("arg$(pos[1])")
				eval(:( $vn = map($c, $vn)))
			end
		end

		par[find(p[2].==1)] = :x  # replace select args with parameter symbol :x for deriv1 testing func
		vn = symbol("arg$(find(p[2].==1)[1])")
		x0 = eval(vn)  # set x0 for deriv 1
		f = expr(:call, [fsym, par...]...) 
		deriv1(f, x0)  # slight shift of x0 to avoid numerical errors on max & min functions
	end
end

## macro to simplify tests expression
macro mtest(pattern, func::Expr, constraints...)
	quote
		local fsym = $(expr(:quote, func.args[1]))
		local pars = $(expr(:quote, [func.args[2:end]...]) ) 
		local rules = $(expr(:quote, [constraints...]) ) 

		combin = ($pattern)(pars, rules)
		runpattern(fsym, pars, rules, combin)
	end
end

#########################################################################
#  tests 
#########################################################################

## regular functions
@mtest testpattern1 x+y
@mtest testpattern1 x+y+z 
@mtest testpattern1 sum(x)
@mtest testpattern1 x-y
@mtest testpattern1 x.*y
@mtest testpattern1 x./y  y -> y==0 ? 0.1 : y
@mtest testpattern1 x.^y  x -> x<=0 ? 0.1 : x  # fails
@mtest testpattern1 sin(x)
@mtest testpattern1 cos(x)
@mtest testpattern1 exp(x)
@mtest testpattern1 log(x) x -> x<=0 ? 0.1 : x
@mtest testpattern1 transpose(x) 
@mtest testpattern1 max(x,y) # fails
@mtest testpattern1 min(x,y) # fails

@mtest testpattern2 x^y 

@mtest testpattern5 x/y   y->y==0 ? 0.1 : y

@mtest testpattern5 x*y 
# add array * array cases

@mtest testpattern2 dot(x,y) 
@mtest testpattern3 dot(x,y) 

## distributions
@mtest testpattern1 SimpleMCMC.logpdfNormal(mu,sigma,x)  sigma -> sigma<=0 ? 0.1 : sigma
@mtest testpattern1 SimpleMCMC.logpdfWeibull(sh,sc,x)    sh->sh<=0?0.1:sh  sc->sc<=0?0.1:sc  x->x<=0?0.1:x
@mtest testpattern1 SimpleMCMC.logpdfBernoulli(prob,x)   prob->min(0.99, max(0.01, prob)) x->(x>0)+0.

@mtest testpattern1 SimpleMCMC.logpdfUniform(a,b,x)      a->a-10 b->b+10


f = prob->min(0.99, max(0.01, prob))
f(-0)

# @mtest SimlpeMCMC.sin(c) (2,2) x>y y<0.

#########################################################################
#  real parameter with other real/vector/matrix arguments 
#########################################################################
# note that we are testing gradient calculations and not full partial derivatives
# => all the expression tested have to evaluate to a scalar (but can depend on vectors)

z = [2., 3, 0.1]
zz = [-1. 3 0 ; 0 5 -2]
zz2 = [-1. 3 0.1 ; 0.1 5 -2]  # version without zeros 
zz3 = [1. 3 0.1 ; 0.1 5 2]  # positive version for testing some parameters of distributions


# Bernoulli distrib
# note : having p=1 or 0 is ok but will make the numeric differentiator of deriv1 fail => not tested

# parsing phase should throw an error from the gradient calc function
try
	@mult deriv1    SimpleMCMC.logpdfBernoulli(1, x)    {-1., 0., 1., 10.}
	throw("no error !!")
catch e
	assert(e != "no error !!", 
		"parser not throwing error when logpdfBernoulli has a parameter dependant sampled variable")
end

@mult deriv1    SimpleMCMC.logpdfBernoulli(x, 1)   {0.999, 0.01, 0.8} 
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, 0)   {0.999, 0., 0.8} 
z2 = [0, 1, 1, 0] # binary outcomes
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, z2)   {0.999, 0.01, 0.8} 
zz4 = [0 1 1 0 ; 1 1 1 0 ; 0 0 1 0] # binary outcomes
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, zz4)   {0.999, 0.01, 0.8} 








#########################################################################
#   vector parameter with other real/vector/matrix arguments 
#########################################################################
tz = transpose(z)

@mult deriv1    sum(x*tz)      {[-3., 2, 0], [1., 10, 8]} 
@mult deriv1    tz*x           {[-3., 2, 0], [1., 10, 8]}  
@mult deriv1    sum(zz*x)      {[-3., 2, 0], [1., 10, 8]}  
@mult deriv1    sum(x*zz[2,:]) {[-3., 2], [1., 10]}  

@mult deriv1    dot(x,z)  {[-3., 2, 0], [1., 10, 8]} 
@mult deriv1    dot(z,x)  {[-3., 2, 0], [1., 10, 8]} 


# Bernoulli distrib
# note : having p=0 and one of the x = 1 is ok but will make the numeric differentiator of deriv1 fail => not tested
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, 0)   {[0., 0.2, 0.5, 0.99], [0., 0.5, 0.99]} 
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, 1)   {[0.2, 0.5, 0.99], [0.5, 0.99]} 
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, z2)   {[0., 0.2, 0.5, 0.99], [0., 0.001, 0.7, 0.8]} 







#########################################################################
# matrix parameter with other real/vector/matrix arguments 
#########################################################################

@mult deriv1    sum(2*x)         {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x*2)         {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x[:,1]*tz)   {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(zz[:,1:2]*x) {[-3. 2 0 ; 1 1 -2]} 
@mult deriv1    sum(x*zz[:,1:2]) {[-3. 2 ; 0 1 ; 1 -2]} 


# Bernoulli distrib
zz4 = [1 0 ; 1 1 ; 0 0 ]
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, 0)   {[0. 0.2 ; 0.5 0.99; 0.2 0.4]} 
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, 1)   {[0.01 0.2 ; 0.5 0.99; 0.2 0.4]}
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, zz4) {[0.01 0.2 ; 0.5 0.99; 0.2 0.4]} 





#####################################################################
#   ref  testing
#####################################################################

@mult deriv1    x[2]                   {[-3., -2, -6], [-1., -10, -8]}
@mult deriv1    sum(x[2:3])            {[-3., -2, -6], [-1., -10, -8]}
@mult deriv1    sum(x[2:end])          {[-3., -2, -6], [-1., -10, -8]}

@mult deriv1    sum(x[2:end])          {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    x[2]                   {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x[2:4])            {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x[:,2])            {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x[1,:])            {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x[2:end,:])        {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x[:,2:end])        {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}

@mult deriv1    x[2]+x[1]              {[-3., -2, -6], [-1., -10, -8]}
@mult deriv1    log(x[2]^2+x[1]^2)     {[-3., -2, -6], [-1., -10, -8]}

# fail case when individual elements of an array are set several times
# FIXME : correct var renaming step in unfold...
# model = :(x::real(3); y=x; y[2] = x[1] ; y ~ TestDiff())

