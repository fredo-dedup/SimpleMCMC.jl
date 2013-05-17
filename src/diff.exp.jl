
# include("src/SimpleMCMC.jl")
# func = :(logpdfBeta(a, b, x::Array))
# diff = :( ( digamma(a+b) - digamma(a) + log(x) ) .* ds )
# dv = :x
macro dfunc(func::Expr, dv::Symbol, diff::Expr)  # dv = :x ; func=:(logpdfBeta(a, b, x::Real))
	argsn = map(e-> isa(e, Symbol) ? e : e.args[1], func.args[2:end])
	index = find(dv .== argsn)[1]

	# change var names in signature and diff expr to x1, x2, x3, ..
	smap = { argsn[i] => symbol("x$i") for i in 1:length(argsn) }
	args2 = SimpleMCMC.substSymbols(func.args[2:end], smap)
	m = SimpleMCMC.MCMCModel()
	m.source = SimpleMCMC.substSymbols(diff, smap)
	SimpleMCMC.unfold!(m)  # unfold for easier optimization later

	# diff function name
	fn = symbol("d_$(func.args[1])_x$index")

	fullf = expr(:(=), expr(:call, fn, args2...), expr(:call, :vcat, map(e->expr(:quote, e), m.exprs)...))
	eval(fullf)
end


@dfunc logpdfBeta(a, b, x::Real)   x   sum( (a-1) ./ x - (b-1) ./ (1-x) ) .* ds
@dfunc logpdfBeta(a, b, x::Real)   x   sum( (a-1) ./ x - (b-1) ./ (1-x) ) .* ds

@dfunc logpdfBeta(a::Real, b, x)   a  sum( digamma(a+b) - digamma(a) + log(x) ) .* ds
@dfunc logpdfBeta(a::Array, b, x)  a  ( digamma(a+b) - digamma(a) + log(x) ) .* ds

@dfunc logpdfBeta(a, b::Real, x)   b   sum( digamma(a+b) - digamma(b) + log(1-x) ) .* ds
@dfunc logpdfBeta(a, b::Array, x)  b  ( digamma(a+b) - digamma(b) + log(1-x) ) .* ds


## common operators
@dfunc +(x::Real, y)     x     sum(ds)
@dfunc +(x::Array, y)    x     +ds
@dfunc +(x, y::Real)     y     sum(ds)
@dfunc +(x, y::Array)    y     +ds

@dfunc -x                x     -ds
@dfunc -(x::Real, y)     x     sum(ds)
@dfunc -(x::Array, y)    x     +ds
@dfunc -(x, y::Real)     y     -sum(ds)
@dfunc -(x, y::Array)    y     -ds

@dfunc sum(x)       x     +ds
@dfunc dot(x, y)    x     y * ds
@dfunc dot(x, y)    y     x * ds

@dfunc log(x)       x     ds ./ x
@dfunc exp(x)       x     exp(x) .* ds

@dfunc sin(x)       x     cos(x) .* ds
@dfunc cos(x)       x     -sin(x) .* ds

@dfunc abs(x)       x     sign(x) .* ds

@dfunc *(x::Real, y)     x     sum([ds .* y])
@dfunc *(x::Array, y)    x     ds * transpose(y)
@dfunc *(x, y::Real)     y     sum([ds .* x])
@dfunc *(x, y::Array)    y     transpose(x) * ds

@dfunc .*(x::Real, y)    x     sum([ds .* y])
@dfunc .*(x::Array, y)   x     ds .* y
@dfunc .*(x, y::Real)    y     sum([ds .* x])
@dfunc .*(x, y::Array)   y     ds .* x

@dfunc ^(x::Real, y::Real)  x     y * x ^ (y-1) * ds # Both args reals
@dfunc ^(x::Real, y::Real)  y     log(x) * x ^ y * ds # Both args reals

@dfunc .^(x::Real, y)    x     sum([y .* x .^ (y-1) .* ds])
@dfunc .^(x::Array, y)   x     y .* x .^ (y-1) .* ds
@dfunc .^(x, y::Real)    y     sum([log(x) .* x .^ y .* ds])
@dfunc .^(x, y::Array)   y     log(x) .* x .^ y .* ds

@dfunc /(x::Real, y)          x     sum([ds ./ y])
@dfunc /(x::Array, y::Real)   x     ds ./ y
@dfunc /(x, y::Real)          y     sum([- x ./ (y .* y) .* ds])
@dfunc /(x::Real, y::Array)   y     - x ./ (y .* y) .* ds

@dfunc ./(x::Real, y)          x     sum([ds ./ y])
@dfunc ./(x::Array, y::Real)   x     ds ./ y
@dfunc ./(x, y::Real)          y     sum([- x ./ (y .* y) .* ds])
@dfunc ./(x::Real, y::Array)   y     - x ./ (y .* y) .* ds

@dfunc max(x::Real, y)    x     sum([x .> y] .* ds)
@dfunc max(x::Array, y)   x     [x .> y] .* ds
@dfunc max(x, y::Real)    y     sum([x .< y] .* ds)
@dfunc max(x, y::Array)   y     [x .< y] .* ds

@dfunc min(x::Real, y)    x     sum([x .< y] .* ds)
@dfunc min(x::Array, y)   x     [x .< y] .* ds
@dfunc min(x, y::Real)    y     sum([x .> y] .* ds)
@dfunc min(x, y::Array)   y     [x .> y] .* ds

@dfunc transpose(x::Real)   x   +ds
@dfunc transpose(x::Array)  x   transpose(ds)

## distributions
# @dfunc logpdfNormal(mu, sigma, x)   mu     ( tmp = [(x - mu) ./ (sigma .^ 2)] * ds ; isa(mu, Real) ? sum(tmp) : tmp) .* ds
# @dfunc logpdfNormal(mu, sigma, x)   sigma  ( tmp = ((x - mu).^2 ./ sigma.^2 - 1.) ./ sigma * ds ; isa(sigma, Real) ? sum(tmp) : tmp) .* ds
# @dfunc logpdfNormal(mu, sigma, x)   x      ( tmp = [(mu - x) ./ (sigma .^ 2)] * ds ; isa(x, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfNormal(mu::Real, sigma, x)         mu     sum([(x - mu) ./ (sigma .^ 2)] * ds) .* ds
@dfunc logpdfNormal(mu::Array, sigma, x)        mu     ([(x - mu) ./ (sigma .^ 2)] * ds) .* ds
@dfunc logpdfNormal(mu, sigma::Real, x)         sigma  sum(((x - mu).^2 ./ sigma.^2 - 1.) ./ sigma * ds) .* ds
@dfunc logpdfNormal(mu, sigma::Array, x)        sigma  (((x - mu).^2 ./ sigma.^2 - 1.) ./ sigma * ds) .* ds
@dfunc logpdfNormal(mu, sigma, x::Real)         x      sum([(mu - x) ./ (sigma .^ 2)] * ds) .* ds
@dfunc logpdfNormal(mu, sigma, x::Array)        x      ([(mu - x) ./ (sigma .^ 2)] * ds) .* ds

# @dfunc logpdfUniform(a, b, x)   a   ( tmp = [a .<= x .<= b] .* (ds ./ (b - a)) ; isa(a, Real) ? sum(tmp) : tmp) .* ds
# @dfunc logpdfUniform(a, b, x)   b   ( tmp = [a .<= x .<= b] .* -(ds ./ (b - a)) ; isa(b, Real) ? sum(tmp) : tmp) .* ds
# @dfunc logpdfUniform(a, b, x)   x   zero(x)
@dfunc logpdfUniform(a::Real, b, x)      a   sum([a .<= x .<= b] .* (ds ./ (b - a))) .* ds
@dfunc logpdfUniform(a::Array, b, x)     a   ([a .<= x .<= b] .* (ds ./ (b - a))) .* ds
@dfunc logpdfUniform(a, b::Real, x)      b   sum([a .<= x .<= b] .* -(ds ./ (b - a))) .* ds
@dfunc logpdfUniform(a, b::Array, x)     b   ([a .<= x .<= b] .* -(ds ./ (b - a))) .* ds
@dfunc logpdfUniform(a, b, x)            x   zero(x)

@dfunc logpdfWeibull(sh, sc, x)   sh  ( tmp = (1. - (x./sc).^sh) .* log(x./sc) + 1./sh * ds ; isa(sh, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfWeibull(sh, sc, x)   sc  ( tmp = ((x./sc).^sh - 1.) .* sh./sc * ds ; isa(sc, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfWeibull(sh, sc, x)   x   ( tmp = ((1. - (x./sc).^sh) .* sh - 1.) ./ x * ds ; isa(x, Real) ? sum(tmp) : tmp) .* ds

@dfunc logpdfBeta(a, b, x)     x     ( tmp = (a-1) ./ x - (b-1) ./ (1-x) ; isa(x, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfBeta(a, b, x)     a     ( tmp = digamma(a+b) - digamma(a) + log(x) ; isa(a, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfBeta(a, b, x)     b     ( tmp = digamma(a+b) - digamma(b) + log(1-x) ; isa(b, Real) ? sum(tmp) : tmp) .* ds

@dfunc logpdfTDist(df, x)    x     ( tmp = -(df+1).*x ./ (df+x.*x) ; isa(x, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfTDist(df, x)    df    (tmp2 = (x.*x + df) ; tmp=( (x.*x-1)./tmp2 + log(df./tmp2) + digamma((df+1)/2) - digamma(df/2) ) / 2 ; isa(df, Real) ? sum(tmp) : tmp) .* ds

@dfunc logpdfExponential(sc, x)   x   (isa(x, Real) ? sum(-1/sc) : -1/sc) .* ds
@dfunc logpdfExponential(sc, x)   sc  (isa(sc, Real) ? sum((x-sc)./(sc.*sc)) : (x-sc)./(sc.*sc)) .* ds

@dfunc logpdfGamma(sh, sc, x)   x   ( tmp = -( sc + x - sh.*sc)./(sc.*x) ;         isa(x, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfGamma(sh, sc, x)   sh  ( tmp = log(x) - log(sc) - digamma(sh) ; isa(sh, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfGamma(sh, sc, x)   sc  ( tmp = (x - sc.*sh) ./ (sc.*sc) ;       isa(sc, Real) ? sum(tmp) : tmp) .* ds

@dfunc logpdfCauchy(mu, sc, x)   x   ( tmp = 2(mu-x) ./ (sc.*sc + (x-mu).*(x-mu)) ;  isa(x, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfCauchy(mu, sc, x)   mu  ( tmp = 2(x-mu) ./ (sc.*sc + (x-mu).*(x-mu)) ;  isa(mu, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfCauchy(mu, sc, x)   sc  ( tmp = ((x-mu).*(x-mu) - sc.*sc) ./ (sc.*(sc.*sc + (x-mu).*(x-mu))) ;  isa(sc, Real) ? sum(tmp) : tmp) .* ds

@dfunc logpdflogNormal(lmu, lsc, x)  x   ( tmp2=lsc.*lsc ; tmp = (lmu - tmp2 - log(x)) ./ (tmp2.*x) ;  isa(x, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdflogNormal(lmu, lsc, x)  lmu ( tmp = (log(x) - lmu) ./ (lsc .* lsc) ;  isa(lmu, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdflogNormal(lmu, lsc, x)  lsc ( tmp2=lsc.*lsc ; tmp = (lmu.*lmu - tmp2 - log(x).*(2lmu-log(x))) ./ (lsc.*tmp2) ;  isa(lsc, Real) ? sum(tmp) : tmp) .* ds

# TODO : find a way to implement multi variate distribs that goes along well with vectorization (Dirichlet, Categorical)
# TODO : other continuous distribs ? : Pareto, Rayleigh, Logistic, Levy, Laplace, Dirichlet, FDist
# TODO : other discrete distribs ? : NegativeBinomial, DiscreteUniform, HyperGeometric, Geometric, Categorical


@dfunc logpdfBernoulli(p, x)    p       ( tmp = 1. ./ (p - (1. - x)) ; isa(p, Real) ? sum(tmp) : tmp) * ds
# Note no derivation on x parameter as it is an integer

@dfunc logpdfBinomial(n, p, x)  p       ( tmp = x ./ p - (n-x) ./ (1 - p) ; isa(p, Real) ? sum(tmp) : tmp) * ds
# Note no derivation on x and n parameters as they are integers

@dfunc logpdfPoisson(lambda, x) lambda  ( tmp = x ./ lambda - 1 ; isa(lambda, Real) ? sum(tmp) : tmp) * ds
# Note no derivation on x parameter as it is an integer


#  fake distribution to test gradient code
@dfunc logpdfTestDiff(x)    x      +ds


## Returns gradient expression of opex
function derive(opex::Expr, index::Integer, dsym::Union(Expr,Symbol))  # opex=:(z^x);index=2;dsym=:y
	vs = opex.args[1+index]
	ds = symbol("$(SimpleMCMC.DERIV_PREFIX)$dsym")
	args = opex.args[2:end]
	
	val = findTypesValuesof(args)
	val = (7, 2.)

	fn = symbol("d_$(opex.args[1])_x$index")

	try
		dexp = eval(expr(:call, fn, val...))
		smap = { symbol("x$i") => args[i] for i in 1:length(args)}
		smap[:ds] = ds
		dexp = SimpleMCMC.substSymbols(dexp, smap)
	catch
		error("[derive] Doesn't know how to derive $opex by argument $vs")
	end

	return :($(symbol("$(DERIV_PREFIX)$vs")) = $(symbol("$(DERIV_PREFIX)$vs")) + $dexp )
end



# function derive(opex::Expr, index::Integer, dsym::Union(Expr,Symbol))  # opex=:(-x);index=1;dsym=:y
# 	op = opex.args[1]  # operator
# 	vs = opex.args[1+index]
# 	ds = symbol("$(DERIV_PREFIX)$dsym")
# 	args = opex.args[2:end]
# 	arity = length(opex.args)-1

# 	# strip out module name (for logpdf...) if present
# 	# if isa(op, Expr) && op.head==:(.) && op.args[1]==:SimpleMCMC 
# 	# 	op = op.args[2].args[1]
# 	# end

# 	if has(rules, (op, index, arity))   # is operator/position defined in rules ?
# 		ex, pars = rules[(op, index, arity)]
# 		smap = { pars[i] => args[i] for i in 1:length(pars)}
# 		smap[:ds] = ds
# 		dexp = substSymbols(ex, smap)
# 	else
# 		error("[derive] Doesn't know how to derive operator $op on parameter $vs")
# 	end


# 	return :($(symbol("$(DERIV_PREFIX)$vs")) += $dexp )
# end
