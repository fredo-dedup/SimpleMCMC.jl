#########################################################################
#    testing script
#########################################################################

include("../src/SimpleMCMC.jl")

DIFF_DELTA = 1e-10
ERROR_THRESHOLD = 1e-2

good_enough(x,y) = isfinite(x) ? (abs(x-y) / max(ERROR_THRESHOLD, abs(x))) < ERROR_THRESHOLD : isequal(x,y) 
good_enough(t::Tuple) = good_enough(t[1], t[2])

############# gradient checking function  ######################
# compares numerical gradient to automated gradient

function deriv1(ex::Expr, x0::Union(Float64, Vector{Float64}, Matrix{Float64})) #  ex= :(sum(2+x)) ; x0 = [2., 3]
	global __beta

	println("testing gradient of $ex at x = $x0")

	nx = length(x0)  # nx=3
	model = expr(:block, nx==1 ? :(x::real) : :(x::real($nx)), :(y = $ex), :(y ~ TestDiff()))

	myf, np = SimpleMCMC.buildFunctionWithGradient(model)
	ex2 = myf.args[2] # use body of function only

	__beta = [x0]
	l0, grad0 = eval(ex2)  

	if nx==1
		__beta += DIFF_DELTA
		l, grad = eval(ex2) 
		gradn = (l-l0)/DIFF_DELTA
	else
		# gradn = grad0 * NaN
		gradn = zeros(nx)
		for i in 1:nx # i = 2
			__beta = [x0]
			__beta[i] += DIFF_DELTA
			l, grad = eval(ex2) 
			gradn[i] = (l-l0)/DIFF_DELTA
		end
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


######### real parameter with other real or vector arguments #######################
# note that we are testing gradient calculations and not full partial derivatives
# => all the expression tested have to evaluate to a scalar (but can depend on vectors)
z = [2., 3, 0.1]

@mult deriv1    2+x       {-3., 0., 1., 10.}
@mult deriv1    x+4       {-3., 0., 1., 10.}
@mult deriv1    sum(z+x)  {-1., 0., 1., 10.}
@mult deriv1    sum(x+z)  {-1., 0., 1., 10.}

@mult deriv1    2-x       {-1., 0., 1., 10.}
@mult deriv1    -x        {-1., 0., 1., 10.}
@mult deriv1    sum(z-x)  {-1., 0., 1., 10.}
@mult deriv1    sum(x-z)  {-1., 0., 1., 10.}

@mult deriv1    2*x      {-1., 0., 1., 10.}
@mult deriv1    x*2      {-1., 0., 1., 10.}
@mult deriv1    sum(x*z) {-1., 0., 1., 10.}
@mult deriv1    sum(z*x) {-1., 0., 1., 10.}

@mult deriv1    x.*2      {-1., 0., 1., 10.}
@mult deriv1    2.*x      {-1., 0., 1., 10.}
@mult deriv1    sum(x.*z) {-1., 0., 1., 10.}
@mult deriv1    sum(z.*x) {-1., 0., 1., 10.}

@mult deriv1    2/x       {-1., 0., 1., 10.}
@mult deriv1    x/2       {-1., 0., 1., 10.}
@mult deriv1    sum(z/x)  {-1., 0., 1., 10.}
@mult deriv1    sum(x/z)  {-1., 0., 1., 10.}

@mult deriv1    2./x      {-1., 0., 1., 10.}
@mult deriv1    x./2      {-1., 0., 1., 10.}
@mult deriv1    sum(x./z) {-1., 0., 1., 10.}
@mult deriv1    sum(z./x) {-1., 0., 1., 10.}

@mult deriv1    2^x      {-1., 0., 1., 10.}
@mult deriv1    x^2      {-1., 0., 1., 10.}

@mult deriv1    2.^x      {-1., 0., 1., 10.}
@mult deriv1    x.^2      {-1., 0., 1., 10.}
@mult deriv1    sum(z.^x) {-1., 0., 1., 10.}
@mult deriv1    sum(x.^z) {-1., 0.1, 1., 10.}

@mult deriv1    sum(x)   {-1., 0., 1., 10.}

@mult deriv1    log(x)         {0., 1., 10.} 
@mult deriv1    sum(log(x*z))  {0., 1., 10.} 

@mult deriv1    exp(x)          {-1., 0., 1., 10.}
@mult deriv1    sum(exp(x+z))   {-1., 0., 1., 10.}

@mult deriv1    sin(x)          {-1., 0., 1., 10.}
@mult deriv1    sum(sin(z/x))   {-1., 0.1, 1., 10.}

@mult deriv1    cos(x)           {-1., 0., 1., 10.}
@mult deriv1    sum(cos(x.^z))   {-1., 0.1, 1., 10.}

## Normal distrib
@mult deriv1    SimpleMCMC.logpdfNormal(1, 2, x)    {-1., 0., 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(-1, x, 0)   {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(x, 4, 10)   {-1., 0., 1., 10.}

@mult deriv1    SimpleMCMC.logpdfNormal(z, 1, x) {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(0, z, x) {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(z, z, x) {-1., 0., 1.}

@mult deriv1    SimpleMCMC.logpdfNormal(0, x, z)    {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(z, x, 0)    {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(z, x, z)    {0.1, 1., 10.}

@mult deriv1    SimpleMCMC.logpdfNormal(x, 1, z)    {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(x, z, 1)    {-1., 0., 1.}
@mult deriv1    SimpleMCMC.logpdfNormal(x, z, 1)    {-1., 0., 1.}

# Weibull distrib
@mult deriv1    SimpleMCMC.logpdfWeibull(1, 2, x)    {0.0001, 1., 10.} # ERROR at 0.0 !!
@mult deriv1    SimpleMCMC.logpdfWeibull(0.5, x, 3)  {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, 4, 10)   {0.1, 1., 10.}

@mult deriv1    SimpleMCMC.logpdfWeibull(z, 1, x)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(2, z, x)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(z, z, x)    {.1, 1., 10.}

@mult deriv1    SimpleMCMC.logpdfWeibull(z, x, 1)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(2, x, z)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(z, x, z)    {.1, 1., 10.}

@mult deriv1    SimpleMCMC.logpdfWeibull(x, z, 1)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, 1, z)    {.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, z, z)    {.1, 1., 10.}

# Uniform distrib
@mult deriv1    SimpleMCMC.logpdfUniform(-1, 1, x)     {-.1, 0., 0.9} 
@mult deriv1    SimpleMCMC.logpdfUniform(0, x, 1.0)    {1.5, 1., 2.99}
@mult deriv1    SimpleMCMC.logpdfUniform(x, 0.5, 0.2)  {0., -1., -60.}

@mult deriv1    SimpleMCMC.logpdfUniform(z, 20, x)     {4., 9., 10.}
@mult deriv1    SimpleMCMC.logpdfUniform(-10, z, x)    {-.1, -1., 0.}
@mult deriv1    SimpleMCMC.logpdfUniform(z, z+5, x)    {3.1, 5., 4.}

@mult deriv1    SimpleMCMC.logpdfUniform(z, x, 4)       {5.1, 9., 10.}
@mult deriv1    SimpleMCMC.logpdfUniform(-5, x, z)      {4.1, 5., 10.}
@mult deriv1    SimpleMCMC.logpdfUniform(z, x, z+1.)    {6.1, 7., 10.}

@mult deriv1    SimpleMCMC.logpdfUniform(x, z, 0)       {-.1, -1., -10.}
@mult deriv1    SimpleMCMC.logpdfUniform(x, 5, z)       {0., -1., -10.}
@mult deriv1    SimpleMCMC.logpdfUniform(x, z, z-1.)    {-2.1, -3., -10.}

# Bernoulli distrib
# note : having p=1 is ok but will make the numeric differentiator of deriv1 fail => not tested

# should throw an error from the gradient calc function
# @mult deriv1    SimpleMCMC.logpdfBernoulli(1, x)    {-1., 0., 1., 10.}

@mult deriv1    SimpleMCMC.logpdfBernoulli(x, 1)   {0.999, 0., 0.8} 
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, 0)   {0.999, 0., 0.8} 
z2 = [0, 1, 1, 0] # binary outcomes
@mult deriv1    SimpleMCMC.logpdfBernoulli(x, z2)   {0.999, 0., 0.8} 


######### vector parameter with other real or vector arguments #######################
tz = transpose(z)

@mult deriv1    sum(2+x)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(x+4)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(z+x)  {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    sum(x+z)  {[-3., 2, 0], [1., 10, 8]}

@mult deriv1    sum(2-x)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(-x)   {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(z-x)  {[-3., 2, 0], [1., 10, 8]}
@mult deriv1    sum(x-z)  {[-3., 2, 0], [1., 10, 8]}

@mult deriv1    sum(2*x)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(x*2)  {[-3., 2, 0], [1., 1.]}
@mult deriv1    sum(x*tz) {[-3., 2, 0], [1., 10, 8]} 
@mult deriv1    tz*x      {[-3., 2, 0], [1., 10, 8]}  

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


############## test refs  ###############################

@mult deriv1    x[2]                   {[-3., -2, -6], [-1., -10, -8]}
@mult deriv1    x[2:3]                 {[-3., -2, -6], [-1., -10, -8]}
@mult deriv1    x[2:end]                 {[-3., -2, -6], [-1., -10, -8]}

@mult deriv1    x[2]+x[1]              {[-3., -2, -6], [-1., -10, -8]}
@mult deriv1    log(x[2]^2+x[1]^2)     {[-3., -2, -6], [-1., -10, -8]}

# fail case when individual elements of an array are set several times
# TODO : correct var renaming step in unfold...
# model = :(x::real(3); y=x; y[2] = x[1] ; y ~ TestDiff())

###### test distributions x samplers  ############################

#  TODO  (find way not to throw error when mean and std slightly off)
# ERROR_THRESHOLD = 1e-1
# model = quote
# 	x::real
	# x ~ Normal(0,1)
# end
# res = SimpleMCMC.simpleRWM(model, 10000)
# assert(good_enough(mean(res[:,3]), 0.0), "Normal mean fail")
# assert(good_enough(std(res[:,3]), 1.0), "Normal std fail")
# x = mean(res[:,3])
# (abs(x) / max(ERROR_THRESHOLD, abs(x))) < ERROR_THRESHOLD : isequal(x,y) 
