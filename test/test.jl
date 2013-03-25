#########################################################################
#    testing script for gradients
#########################################################################

include("../src/SimpleMCMC.jl")

DIFF_DELTA = 1e-10
ERROR_THRESHOLD = 1e-2

good_enough(x,y) = isfinite(x) ? (abs(x-y) / max(ERROR_THRESHOLD, abs(x))) < ERROR_THRESHOLD : isequal(x,y) 
good_enough(t::Tuple) = good_enough(t[1], t[2])

############# gradient checking function  ######################
# compares numerical gradient to automated gradient

function deriv1(ex::Expr, x0::Union(Float64, Vector{Float64}, Matrix{Float64})) #  ex= :(sum(2+x)) ; x0 = [2., 3]
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






#########################################################################
#  real parameter with other real/vector/matrix arguments 
#########################################################################
# note that we are testing gradient calculations and not full partial derivatives
# => all the expression tested have to evaluate to a scalar (but can depend on vectors)

z = [2., 3, 0.1]
zz = [-1. 3 0 ; 0 5 -2]
zz2 = [-1. 3 0.1 ; 0.1 5 -2]  # version without zeros 
zz3 = [1. 3 0.1 ; 0.1 5 2]  # positive version for testing some parameters of distributions

@mult deriv1    2+x       {-3., 0., 1., 10.}
@mult deriv1    x+4       {-3., 0., 1., 10.}
@mult deriv1    sum(z+x)  {-1., 0., 1., 10.}
@mult deriv1    sum(x+z)  {-1., 0., 1., 10.}
@mult deriv1    sum(zz+x)  {-1., 0., 1., 10.}
@mult deriv1    sum(x+zz)  {-1., 0., 1., 10.}

@mult deriv1    2-x       {-1., 0., 1., 10.}
@mult deriv1    -x        {-1., 0., 1., 10.}
@mult deriv1    sum(z-x)  {-1., 0., 1., 10.}
@mult deriv1    sum(x-z)  {-1., 0., 1., 10.}
@mult deriv1    sum(zz-x)  {-1., 0., 1., 10.}
@mult deriv1    sum(x-zz)  {-1., 0., 1., 10.}

@mult deriv1    2*x      {-1., 0., 1., 10.}
@mult deriv1    x*2      {-1., 0., 1., 10.}
@mult deriv1    sum(x*z) {-1., 0., 1., 10.}
@mult deriv1    sum(z*x) {-1., 0., 1., 10.}
@mult deriv1    sum(x*zz) {-1., 0., 1., 10.}  
@mult deriv1    sum(zz*x) {-1., 0., 1., 10.}  

@mult deriv1    x.*2      {-1., 0., 1., 10.}
@mult deriv1    2.*x      {-1., 0., 1., 10.}
@mult deriv1    sum(x.*z) {-1., 0., 1., 10.}
@mult deriv1    sum(z.*x) {-1., 0., 1., 10.}
@mult deriv1    sum(x.*zz) {-1., 0., 1., 10.}  
@mult deriv1    sum(zz.*x) {-1., 0., 1., 10.}  

@mult deriv1    2/x       {-1., 0., 1., 10.}
@mult deriv1    x/2       {-1., 0., 1., 10.}
@mult deriv1    sum(z/x)  {-1., 0., 1., 10.}
@mult deriv1    sum(x/z)  {-1., 0., 1., 10.}
@mult deriv1    sum(zz/x)  {-1., 0., 1., 10.}
@mult deriv1    sum(x/zz2)  {-1., 0., 1., 10.} 

@mult deriv1    2./x      {-1., 0., 1., 10.}
@mult deriv1    x./2      {-1., 0., 1., 10.}
@mult deriv1    sum(z./x) {-1., 0., 1., 10.}
@mult deriv1    sum(x./z) {-1., 0., 1., 10.}
@mult deriv1    sum(zz./x) {-1., 0., 1., 10.}
@mult deriv1    sum(x./zz2) {-1., 0., 1., 10.} 

@mult deriv1    2^x      {-1., 0., 1., 10.}
@mult deriv1    x^2      {-1., 0., 1., 10.}

@mult deriv1    2.^x        {-1., 0., 1., 10.}
@mult deriv1    x.^2        {-1., 0., 1., 10.}
@mult deriv1    sum(z.^x)   {-1., 0., 1., 10.}
@mult deriv1    sum(x.^z)   {-1., 0.1, 1., 10.}
@mult deriv1    sum(zz3.^x) {-1., 0., 1., 10.} 
@mult deriv1    sum(x.^zz)  {-1., 0.1, 1., 10.}

@mult deriv1    sum(x)   {-1., 0., 1., 10.}

@mult deriv1    log(x)         {0., 1., 10.} 
@mult deriv1    sum(log(x*z))  {0., 1., 10.} 
@mult deriv1    sum(log(x*zz3))  {0., 1., 10.} 

@mult deriv1    exp(x)          {-1., 0., 1., 10.}
@mult deriv1    sum(exp(x+z))   {-1., 0., 1., 10.}
@mult deriv1    sum(exp(x+zz))   {-1., 0., 1., 10.}

@mult deriv1    sin(x)          {-1., 0., 1., 10.}
@mult deriv1    sum(sin(z/x))   {-1., 0.1, 1., 10.}
@mult deriv1    sum(sin(zz/x))   {-1., 0.1, 1., 10.}

@mult deriv1    cos(x)           {-1., 0., 1., 10.}
@mult deriv1    sum(cos(x.^z))   {-1., 0.1, 1., 10.}
@mult deriv1    sum(cos(x.^zz))   {-1., 0.1, 1., 10.}

## Normal distrib
# x as sampled var
@mult deriv1    SimpleMCMC.logpdfNormal(1, 2, x)    {-1., 0., 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(z, 1, x) {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(0, z, x) {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(z, z, x) {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(zz, 1, x) {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(0, zz3, x) {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(zz, zz3, x) {-1., 0., 1.}

# x as sigma
@mult deriv1    SimpleMCMC.logpdfNormal(-1, x, 0)   {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(0, x, z)    {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(z, x, 0)    {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(z, x, z)    {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(0, x, zz)    {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(zz, x, 0)    {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(zz, x, zz)    {0.1, 1., 10.}

# x as mu
@mult deriv1    SimpleMCMC.logpdfNormal(x, 4, 10)   {-1., 0., 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(x, 1, z)    {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(x, z, 1)    {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(x, z, 1)    {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(x, 1, z)    {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(x, z, 1)    {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(x, z, 1)    {-1., 0., 1.}

## Weibull distrib
# x as sampled var
@mult deriv1    SimpleMCMC.logpdfWeibull(1, 2, x)    {0.0001, 1., 10.} # ERROR at 0.0 !!
@mult deriv1    SimpleMCMC.logpdfWeibull(z, 1, x)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(2, z, x)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(z, z, x)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(zz3, 1, x)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(2, zz3, x)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(zz3, zz3, x)    {.1, 1., 10.}

# x as scale param
@mult deriv1    SimpleMCMC.logpdfWeibull(0.5, x, 3)  {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(z, x, 1)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(2, x, z)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(z, x, z)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(zz3, x, 1)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(2, x, zz3)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(zz3, x, zz3)    {.1, 1., 10.}

# x as shape param
@mult deriv1    SimpleMCMC.logpdfWeibull(x, 4, 10)   {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, z, 1)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, 1, z)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, z, z)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, zz3, 1)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, 1, zz3)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, zz3, zz3)    {.1, 1., 10.}

## Uniform distrib
# x as sampled var
@mult deriv1    SimpleMCMC.logpdfUniform(-1, 1, x)     {-.1, 0., 0.9} 
@mult deriv1    SimpleMCMC.logpdfUniform(z, 20, x)     {4., 9., 10.}
@mult deriv1    SimpleMCMC.logpdfUniform(-10, z, x)    {-.1, -1., 0.}
@mult deriv1    SimpleMCMC.logpdfUniform(z, z+5, x)    {3.1, 5., 4.}
@mult deriv1    SimpleMCMC.logpdfUniform(zz, 20, x)     {5., 9., 10.}
@mult deriv1    SimpleMCMC.logpdfUniform(-10, zz3, x)    {-.1, -1., 0.}
@mult deriv1    SimpleMCMC.logpdfUniform(zz, zz+10, x)    {5.1, 6., 7.}

# x as b param
@mult deriv1    SimpleMCMC.logpdfUniform(0, x, 1.0)    {1.5, 1., 2.99}
@mult deriv1    SimpleMCMC.logpdfUniform(z, x, 4)       {5.1, 9., 10.}
@mult deriv1    SimpleMCMC.logpdfUniform(-5, x, z)      {4.1, 5., 10.}
@mult deriv1    SimpleMCMC.logpdfUniform(z, x, z+1.)    {6.1, 7., 10.}
@mult deriv1    SimpleMCMC.logpdfUniform(zz, x, 6)       {7.1, 9., 10.}
@mult deriv1    SimpleMCMC.logpdfUniform(-5, x, zz)      {6.1, 30., 12.1}
@mult deriv1    SimpleMCMC.logpdfUniform(zz, x, zz+1.)    {6.1, 7., 10.}

# x as a param
@mult deriv1    SimpleMCMC.logpdfUniform(x, 0.5, 0.2)  {0., -1., -60.}
@mult deriv1    SimpleMCMC.logpdfUniform(x, z, 0)       {-.1, -1., -10.}
@mult deriv1    SimpleMCMC.logpdfUniform(x, 5, z)       {0., -1., -10.}
@mult deriv1    SimpleMCMC.logpdfUniform(x, z, z-1.)    {-2.1, -3., -10.}
@mult deriv1    SimpleMCMC.logpdfUniform(x, zz, -3)       {-6.1, -11., -10.}
@mult deriv1    SimpleMCMC.logpdfUniform(x, 6, zz)       {-10., -3., -2.1}
@mult deriv1    SimpleMCMC.logpdfUniform(x, zz, zz-.1)    {-2.2, -3., -10.}

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

@mult deriv1    sum(2+x)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(x+4)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(z+x)  {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    sum(x+z)  {[-3., 2, 0], [1., 10, 8]}

@mult deriv1    sum(2-x)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(-x)   {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(z-x)  {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    sum(x-z)  {[-3., 2, 0], [1., 10, 8]}

@mult deriv1    sum(2*x)       {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(x*2)       {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(x*tz)      {[-3., 2, 0], [1., 10, 8]} 
@mult deriv1    tz*x           {[-3., 2, 0], [1., 10, 8]}  
@mult deriv1    sum(zz*x)      {[-3., 2, 0], [1., 10, 8]}  
@mult deriv1    sum(x*zz[2,:]) {[-3., 2], [1., 10]}  

@mult deriv1    sum(x.*2) {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(2.*x) {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(x.*z) {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    sum(z.*x) {[-3., 2, 0], [1., 10, 8]}

@mult deriv1    dot(x,z)  {[-3., 2, 0], [1., 10, 8]} 
@mult deriv1    dot(z,x)  {[-3., 2, 0], [1., 10, 8]} 

@mult deriv1    sum(2/x)  {[-3., 2, 0.01], [1., 1.]}  
@mult deriv1    sum(x/2)  {[-3., 2, 0], [1., 1.]}

@mult deriv1    sum(2./x) {[-3., 2, 0.1], [1., 1.]} 
@mult deriv1    sum(x./2) {[-3., 2, 0.1], [1., 1.]}
@mult deriv1    sum(x./z) {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    sum(z./x) {[-3., 2, 0.1], [1., 10, 8]}  

@mult deriv1    sum(2.^x)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(x.^2)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(z.^x)  {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    sum(x.^z)  {[-3., 2, 0.1], [1., 10, 1]} 

@mult deriv1    sum(x)     {[-3., 2, 0], [1., 1.]}

@mult deriv1    sum(log(x))     {[3., 2, 0.1], [1., 1.]} 

@mult deriv1    sum(exp(x))     {[-3., 2, 0], [1., 1.]}

@mult deriv1    sum(sin(x))     {[-3., 2, 0], [1., 1.]}

@mult deriv1    sum(cos(x))     {[-3., 2, 0], [1., 1.]}

## Normal distrib
@mult deriv1    SimpleMCMC.logpdfNormal(1, 2, x)    {[-3., 2, 0], [1., 1.]}
@mult deriv1    SimpleMCMC.logpdfNormal(-1, x, 0)   {[3., 2, 0.1], [1., 1.]} 
@mult deriv1    SimpleMCMC.logpdfNormal(x, 4, 10)   {[-3., 2, 0], [1., 1.]}

@mult deriv1    SimpleMCMC.logpdfNormal(z, 1, x)    {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    SimpleMCMC.logpdfNormal(0, z, x)    {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    SimpleMCMC.logpdfNormal(z, z, x)    {[-3., 2, 0], [1., 10, 8]}

@mult deriv1    SimpleMCMC.logpdfNormal(0, x, z)    {[3., 2, 0.1], [1., 10, 8]} 
@mult deriv1    SimpleMCMC.logpdfNormal(z, x, 0)    {[3., 2, 0.1], [1., 10, 8]} 
@mult deriv1    SimpleMCMC.logpdfNormal(z, x, z)    {[3., 2, 0.1], [1., 10, 8]} 

@mult deriv1    SimpleMCMC.logpdfNormal(x, 1, z)    {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    SimpleMCMC.logpdfNormal(x, z, 1)    {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    SimpleMCMC.logpdfNormal(x, z, 1)    {[-3., 2, 0], [1., 10, 8]}

# Weibull distrib
@mult deriv1    SimpleMCMC.logpdfWeibull(1, 2, x)    {[3., 2, 0.1], [1., 1.]}
@mult deriv1    SimpleMCMC.logpdfWeibull(0.5, x, 3)  {[3., 2, 0.1], [1., 1.]}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, 4, 10)   {[3., 2, 0.1], [1., 1.]}

@mult deriv1    SimpleMCMC.logpdfWeibull(z, 1, x)    {[3., 2, 0.1], [1., 10, 8]}
@mult deriv1    SimpleMCMC.logpdfWeibull(2, z, x)    {[3., 2, 0.1], [1., 10, 8]}
@mult deriv1    SimpleMCMC.logpdfWeibull(z, z, x)    {[3., 2, 0.1], [1., 10, 8]}

@mult deriv1    SimpleMCMC.logpdfWeibull(z, x, 1)    {[3., 2, 0.1], [1., 5, 8]}
@mult deriv1    SimpleMCMC.logpdfWeibull(2, x, z)    {[3., 2, 0.1], [1., 5, 8]}
@mult deriv1    SimpleMCMC.logpdfWeibull(z, x, z)    {[3., 2, 0.1], [1., 5, 8]}

@mult deriv1    SimpleMCMC.logpdfWeibull(x, z, 1)    {[3., 2, 0.1], [1., 2, 3]}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, 1, z)    {[3., 2, 0.1], [1., 2, 3]}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, z, z)    {[3., 2, 0.1], [1., 2, 3]}

# Uniform distrib
@mult deriv1    SimpleMCMC.logpdfUniform(-5, 2.2, x)   {[-3., 2, 0], [1., 1.]}
@mult deriv1    SimpleMCMC.logpdfUniform(-5, x, -4.0)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    SimpleMCMC.logpdfUniform(x, 4.5, 2.2)  {[-3., 2, 0], [1., 1.]}

@mult deriv1    SimpleMCMC.logpdfUniform(z, 20, x)     {[3.1, 4.2, 5], [5., 10, 8]}
@mult deriv1    SimpleMCMC.logpdfUniform(-10, z, x)    {[-3., -2, 0], [-1., -10, -8]}
@mult deriv1    SimpleMCMC.logpdfUniform(z, z+5, x)    {[3.1, 4.2, 5]}

@mult deriv1    SimpleMCMC.logpdfUniform(z, x, 6)      {[6.1, 10.2, 7], [9., 10, 8]}
@mult deriv1    SimpleMCMC.logpdfUniform(-5, x, z)     {[3.5, 4.2, 8], [7., 10, 8]}
@mult deriv1    SimpleMCMC.logpdfUniform(z, x, z+1.)   {[6.1, 10.2, 7], [9., 10, 8]}

@mult deriv1    SimpleMCMC.logpdfUniform(x, z, 0)      {[-3., -2, -6], [-1., -10, -8]}
@mult deriv1    SimpleMCMC.logpdfUniform(x, 5, z)      {[-3., -2, -6], [-1., -10, -8]}
@mult deriv1    SimpleMCMC.logpdfUniform(x, z, z-1.)   {[-3., -2, -6], [-1., -10, -8]}

# Bernoulli distrib
# note : having p=0 and one of the x = 1 is ok but will make the numeric differentiator of deriv1 fail => not tested
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, 0)   {[0., 0.2, 0.5, 0.99], [0., 0.5, 0.99]} 
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, 1)   {[0.2, 0.5, 0.99], [0.5, 0.99]} 
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, z2)   {[0., 0.2, 0.5, 0.99], [0., 0.001, 0.7, 0.8]} 







#########################################################################
# matrix parameter with other real/vector/matrix arguments 
#########################################################################

@mult deriv1    sum(2+x)  {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x+4)  {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(zz+x)  {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    sum(x+zz)  {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    sum(x[:,2] + z)  {[-3. 2 ; 0 1 ; 1 -2]} 

@mult deriv1    sum(2-x)  {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(-x)   {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(zz-x)  {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    sum(x-zz3)  {[-3. 2 0 ; 1 1 -2]}

@mult deriv1    sum(2*x)         {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x*2)         {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x[:,1]*tz)   {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(zz[:,1:2]*x) {[-3. 2 0 ; 1 1 -2]} 
@mult deriv1    sum(x*zz[:,1:2]) {[-3. 2 ; 0 1 ; 1 -2]} 

@mult deriv1    sum(x.*2) {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(2.*x) {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}
@mult deriv1    sum(x.*zz) {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    sum(zz.*x) {[-3. 2 0 ; 1 1 -2]}

@mult deriv1    sum(2/x)  {[-3. 2 0.1 ; 1 1 -2], [-3. 2 ; -0.2 1 ; 1 -2]}
@mult deriv1    sum(x/2)  {[-3. 2 0 ; 1 1 -2], [-3. 2 ; 0 1 ; 1 -2]}

@mult deriv1    sum(2./x)   {[-3. 2 0.1 ; 1 1 -2], [-3. 2 ; -0.2 1 ; 1 -2]}
@mult deriv1    sum(x./2)   {[-3. 2 0.1 ; 1 1 -2], [-3. 2 ; -0.2 1 ; 1 -2]}
@mult deriv1    sum(x./zz2) {[-3. 2 0.1 ; 1 1 -2]}
@mult deriv1    sum(zz./x)   {[-3. 2 0.1 ; 1 1 -2]} 

@mult deriv1    sum(2.^x)     {[-3. 2 0.1 ; 1 1 -2], [-3. 2 ; -0.2 1 ; 1 -2]}
@mult deriv1    sum(x.^2)     {[-3. 2 0.1 ; 1 1 -2], [-3. 2 ; -0.2 1 ; 1 -2]}
@mult deriv1    sum(zz3.^x)   {[-3. 2 0.1 ; 1 1 -2]} 
@mult deriv1    sum(x.^zz2)   {[-3. 2 0.1 ; 1 1 -2]} 

@mult deriv1    sum(x)     {[-3. 2 0.1 ; 1 1 -2], [-3. 2 ; -0.2 1 ; 1 -2]}

@mult deriv1    sum(log(x))     {[3. 2 0.1 ; 1 1 2], [3. 2 ; 0.2 1 ; 1 2]}

@mult deriv1    sum(exp(x))     {[-3. 2 0.1 ; 1 1 -2], [-3. 2 ; -0.2 1 ; 1 -2]}

@mult deriv1    sum(sin(x))     {[-3. 2 0.1 ; 1 1 -2], [-3. 2 ; -0.2 1 ; 1 -2]}

@mult deriv1    sum(cos(x))     {[-3. 2 0.1 ; 1 1 -2], [-3. 2 ; -0.2 1 ; 1 -2]}

## Normal distrib
# x as sampled var
@mult deriv1    SimpleMCMC.logpdfNormal(1, 2, x)     {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    SimpleMCMC.logpdfNormal(zz, 1, x)    {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    SimpleMCMC.logpdfNormal(0, zz3, x)   {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    SimpleMCMC.logpdfNormal(zz, zz3, x)  {[-3. 2 0 ; 1 1 -2]}

# x as sigma
@mult deriv1    SimpleMCMC.logpdfNormal(-1, x, 0)   {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfNormal(0, x, zz)   {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfNormal(zz, x, 0)   {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfNormal(zz, x, zz)  {[3. 2 0.1 ; 1 1 2]}

# x as mu
@mult deriv1    SimpleMCMC.logpdfNormal(x, 4, 10)   {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    SimpleMCMC.logpdfNormal(x, 1, zz)   {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    SimpleMCMC.logpdfNormal(x, zz3, 1)  {[-3. 2 0 ; 1 1 -2]}
@mult deriv1    SimpleMCMC.logpdfNormal(x, zz3, zz) {[-3. 2 0 ; 1 1 -2]}

## Weibull distrib
# x as sampled var
@mult deriv1    SimpleMCMC.logpdfWeibull(1, 2, x)     {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfWeibull(zz3, 1, x)   {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfWeibull(2, zz3, x)   {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfWeibull(zz3, zz3, x) {[3. 2 0.1 ; 1 1 2]}

# x as scale param
@mult deriv1    SimpleMCMC.logpdfWeibull(0.5, x, 3)    {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfWeibull(zz3, x, 1)    {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfWeibull(2, x, zz3)    {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfWeibull(zz3, x, zz3)  {[3. 1 0.1 ; 0.1 1 2]}

# x as shape param
@mult deriv1    SimpleMCMC.logpdfWeibull(x, 4, 10)    {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, zz3, 1)   {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, 1, zz3)   {[3. 2 0.1 ; 1 1 2]}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, zz3, zz3) {[3. 2 0.1 ; 1 1 2]}

## Uniform distrib
# x as sampled var
@mult deriv1    SimpleMCMC.logpdfUniform(-4, 6, x)      {zz}
@mult deriv1    SimpleMCMC.logpdfUniform(zz, 7, x)      {zz+1}
@mult deriv1    SimpleMCMC.logpdfUniform(-10, zz+1, x)  {zz}
@mult deriv1    SimpleMCMC.logpdfUniform(zz, zz+2, x)   {zz}

# x as b param
@mult deriv1    SimpleMCMC.logpdfUniform(-3, x, -2.5)    {zz}
@mult deriv1    SimpleMCMC.logpdfUniform(zz, x, 6)       {zz+9}
@mult deriv1    SimpleMCMC.logpdfUniform(-5, x, zz)      {zz+1}
@mult deriv1    SimpleMCMC.logpdfUniform(zz, x, zz+1.)   {zz+2}

# x as a param
@mult deriv1    SimpleMCMC.logpdfUniform(x, 7, 6)       {zz}
@mult deriv1    SimpleMCMC.logpdfUniform(x, zz+8, 6)    {zz}
@mult deriv1    SimpleMCMC.logpdfUniform(x, 7, zz+1)    {zz}
@mult deriv1    SimpleMCMC.logpdfUniform(x, zz+2, zz+1) {zz}

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

