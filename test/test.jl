#########################################################################
#    testing script
#########################################################################

include("../src/SimpleMCMC.jl")

DIFF_DELTA = 1e-10
ERROR_THRESHOLD = 1e-2

############# gradient checking function  ######################

function deriv1(ex::Expr, x0::Union(Float64, Vector{Float64}, Matrix{Float64})) #  ex= :(2*x) ; x0 = 2.
	println("testing derivation of $ex at x = $x0")

	model = :(x::real; y = $ex; y ~ TestDiff())

	myf, np = SimpleMCMC.buildFunctionWithGradient(model)
	ex2 = myf.args[2].args

	cex = expr(:block, vcat({:(__beta = [$x0])}, ex2))
	l0, grad0 = eval(cex)  

	cex = expr(:block, vcat({:(__beta = [$(x0+DIFF_DELTA)])}, ex2))
	l, grad = eval(cex) 
	gradn = (l-l0)/DIFF_DELTA

	good_enough(x,y) = isfinite(x) ? (abs(x-y) / max(0.01, abs(x))) < ERROR_THRESHOLD : isequal(x,y) 
	good_enough(t::Tuple) = good_enough(t[1], t[2])

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


######### real var derivation with other real arguments #######################

@mult deriv1    2+x     {-3., 0., 1., 10.}
@mult deriv1    x+4     {-3., 0., 1., 10.}

@mult deriv1    2-x     {-1., 0., 1., 10.}
@mult deriv1    -x      {-1., 0., 1., 10.}

@mult deriv1    2*x      {-1., 0., 1., 10.}
@mult deriv1    x*2      {-1., 0., 1., 10.}

@mult deriv1    x.*2   {-1., 0., 1., 10.}
@mult deriv1    2.*x   {-1., 0., 1., 10.}

@mult deriv1    2/x      {-1., 0., 1., 10.}
@mult deriv1    x/2      {-1., 0., 1., 10.}

@mult deriv1    2./x      {-1., 0., 1., 10.}
@mult deriv1    x./2      {-1., 0., 1., 10.}

@mult deriv1    2^x      {-1., 0., 1., 10.}
@mult deriv1    x^2      {-1., 0., 1., 10.}

@mult deriv1    2.^x      {-1., 0., 1., 10.}
@mult deriv1    x.^2      {-1., 0., 1., 10.}

@mult deriv1    sum(x)   {-1., 0., 1., 10.}

@mult deriv1    log(x)   {0., 1., 10.} # TODO : make deriv = NaN when x < 0
@mult deriv1    exp(x)   {-1., 0., 1., 10.}
@mult deriv1    sin(x)   {-1., 0., 1., 10.}
@mult deriv1    cos(x)   {-1., 0., 1., 10.}

@mult deriv1    SimpleMCMC.logpdfNormal(1, 2, x)    {-1., 0., 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(-1, x, 0)   {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfNormal(x, 4, 10)   {-1., 0., 1., 10.}

# @mult deriv1    SimpleMCMC.logpdfWeibull(1, 2, x)    {0., 1., 10.} # ERROR + TODO : make deriv = NaN when x < 0
@mult deriv1    SimpleMCMC.logpdfWeibull(0.5, x, 3)  {0.1, 1., 10.}
@mult deriv1    SimpleMCMC.logpdfWeibull(x, 4, 10)   {0.1, 1., 10.}

@mult deriv1    SimpleMCMC.logpdfUniform(-1, 1, x)     {-.1, 0., 0.9} # TODO : make deriv = NaN when x < a or > b
@mult deriv1    SimpleMCMC.logpdfUniform(0, x, 1.0)    {1.5, 1., 2.99}
@mult deriv1    SimpleMCMC.logpdfUniform(x, 0.5, 0.2)  {0., -1., -60.}

######### real var derivation with other vector arguments #######################
z = [2., 3, 0.1]

@mult deriv1 z+x {-1., 0., 1., 10.}
@mult deriv1 x+z {-1., 0., 1., 10.}

@mult deriv1 z-x {-1., 0., 1., 10.}
@mult deriv1 x-z {-1., 0., 1., 10.}

@mult deriv1 sum(x*z) {-1., 0., 1., 10.}  # note that 'sum' is necessary here because result has to be a scalar
@mult deriv1 sum(z*x) {-1., 0., 1., 10.}

@mult deriv1 sum(x.*z) {-1., 0., 1., 10.}
@mult deriv1 sum(z.*x) {-1., 0., 1., 10.}

@mult deriv1 z/x {-1., 0., 1., 10.}
@mult deriv1 x/z {-1., 0., 1., 10.}

@mult deriv1 sum(x./z) {-1., 0., 1., 10.}
@mult deriv1 sum(z./x) {-1., 0., 1., 10.}


@mult deriv1 SimpleMCMC.logpdfNormal(z, 1, x) {-1., 0., 1.}
@mult deriv1 SimpleMCMC.logpdfNormal(0, z, x) {-1., 0., 1.}
@mult deriv1 SimpleMCMC.logpdfNormal(z, z, x) {-1., 0., 1.}

@mult deriv1 SimpleMCMC.logpdfNormal(0, x, z)    {0.1, 1., 10.}
@mult deriv1 SimpleMCMC.logpdfNormal(z, x, 0)    {0.1, 1., 10.}
@mult deriv1 SimpleMCMC.logpdfNormal(z, x, z)    {0.1, 1., 10.}

@mult deriv1 SimpleMCMC.logpdfNormal(x, 1, z)    {-1., 0., 1.}
@mult deriv1 SimpleMCMC.logpdfNormal(x, z, 1)    {-1., 0., 1.}
@mult deriv1 SimpleMCMC.logpdfNormal(x, z, 1)    {-1., 0., 1.}


@mult deriv1 SimpleMCMC.logpdfWeibull(z, 1, x)    {.1, 1., 10.}
@mult deriv1 SimpleMCMC.logpdfWeibull(2, z, x)    {.1, 1., 10.}
@mult deriv1 SimpleMCMC.logpdfWeibull(z, z, x)    {.1, 1., 10.}

@mult deriv1 SimpleMCMC.logpdfWeibull(z, x, 1)    {.1, 1., 10.}
@mult deriv1 SimpleMCMC.logpdfWeibull(2, x, z)    {.1, 1., 10.}
@mult deriv1 SimpleMCMC.logpdfWeibull(z, x, z)    {.1, 1., 10.}

@mult deriv1 SimpleMCMC.logpdfWeibull(x, z, 1)    {.1, 1., 10.}
@mult deriv1 SimpleMCMC.logpdfWeibull(x, 1, z)    {.1, 1., 10.}
@mult deriv1 SimpleMCMC.logpdfWeibull(x, z, z)    {.1, 1., 10.}


@mult deriv1 SimpleMCMC.logpdfUniform(z, 20, x)    {4., 9., 10.}
@mult deriv1 SimpleMCMC.logpdfUniform(-10, z, x)    {-.1, -1., 0.}
@mult deriv1 SimpleMCMC.logpdfUniform(z, z+5, x)    {3.1, 5., 4.}

@mult deriv1 SimpleMCMC.logpdfUniform(z, x, 4)    {5.1, 9., 10.}
@mult deriv1 SimpleMCMC.logpdfUniform(-5, x, z)    {4.1, 5., 10.}
@mult deriv1 SimpleMCMC.logpdfUniform(z, x, z+1.)    {6.1, 7., 10.}

@mult deriv1 SimpleMCMC.logpdfUniform(x, z, 0)    {-.1, -1., -10.}
@mult deriv1 SimpleMCMC.logpdfUniform(x, 5, z)    {0., -1., -10.}
@mult deriv1 SimpleMCMC.logpdfUniform(x, z, z-1.)    {-2.1, -3., -10.}


######### vector var derivation with other real arguments #######################




