##########################################################################################
#
#    function 'derive' returning the expr of gradient
#
# TODO : add operators : hcat, vcat, ? : , map, mapreduce, if else 
#
##########################################################################################


## macro to simplify derivation rules creation
macro dfunc(func::Expr, dv::Symbol, diff::Expr)
	quote
		local fsym = $(expr(:quote, func.args[1]))
		local fdiff = $(expr(:quote, diff))
		local pos = $(find(dv .== func.args[2:end])[1])
		local pars = $(expr(:quote, [func.args[2:end]...]) ) 
		local arity = $(length(func.args)-1)

		rules[(fsym, pos, arity)] = {fdiff, pars}
	end
end

## derivation rules
rules = Dict()

## common operators
@dfunc x+y         x     isa(x, Real) ? sum(ds) : ds
@dfunc x+y         y     isa(y, Real) ? sum(ds) : ds
@dfunc -x          x     -ds
@dfunc x-y         x     isa(x, Real) ? sum(ds) : ds
@dfunc x-y         y     isa(y, Real) ? -sum(ds) : -ds

@dfunc sum(x)      x     +ds
@dfunc dot(x,y)    x     y * ds
@dfunc dot(x,y)    y     x * ds

@dfunc log(x)      x     ds ./ x
@dfunc exp(x)      x     exp(x) .* ds

@dfunc sin(x)      x     cos(x) .* ds
@dfunc cos(x)      x     -sin(x) .* ds

@dfunc abs(x)      x     sign(x) .* ds

@dfunc x*y         x     isa(x, Real) ? sum([ds .* y]) : ds * transpose(y)
@dfunc x*y         y     isa(y, Real) ? sum([ds .* x]) : transpose(x) * ds

@dfunc x.*y        x     isa(x, Real) ? sum([ds .* y]) : ds .* y
@dfunc x.*y        y     isa(y, Real) ? sum([ds .* x]) : ds .* x

@dfunc x^y         x     y * x ^ (y-1) * ds # Both args reals
@dfunc x^y         y     log(x) * x ^ y * ds # Both args reals

@dfunc x.^y        x     isa(x, Real) ? sum([y .* x .^ (y-1) .* ds]) : y .* x .^ (y-1) .* ds
@dfunc x.^y        y     isa(y, Real) ? sum([log(x) .* x .^ y .* ds]) : log(x) .* x .^ y .* ds

# FIXME : this will fail silently if both args are arrays
@dfunc x/y         x     isa(x, Real) ? sum([ds ./ y]) : ds ./ y
@dfunc x/y         y     isa(y, Real) ? sum([- x ./ (y .* y) .* ds]) : - x ./ (y .* y) .* ds

@dfunc x./y        x     isa(x, Real) ? sum([ds ./ y]) : ds ./ y
@dfunc x./y        y     isa(y, Real) ? sum([- x ./ (y .* y) .* ds]) : - x ./ (y .* y) .* ds

@dfunc max(x, y)   x     isa(x, Real) ? sum([x .> y] .* ds) : [x .> y] .* ds
@dfunc max(x, y)   y     isa(y, Real) ? sum([x .< y] .* ds) : [x .< y] .* ds

@dfunc min(x, y)   x     isa(x, Real) ? sum([x .< y] .* ds) : [x .< y] .* ds
@dfunc min(x, y)   y     isa(y, Real) ? sum([x .> y] .* ds) : [x .> y] .* ds

@dfunc transpose(x)  x   isa(x, Real) ? ds : transpose(ds)

## distributions
@dfunc logpdfNormal(mu, sigma, x)   mu     ( tmp = [(x - mu) ./ (sigma .^ 2)] * ds ; isa(mu, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfNormal(mu, sigma, x)   sigma  ( tmp = ((x - mu).^2 ./ sigma.^2 - 1.) ./ sigma * ds ; isa(sigma, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfNormal(mu, sigma, x)   x      ( tmp = [(mu - x) ./ (sigma .^ 2)] * ds ; isa(x, Real) ? sum(tmp) : tmp) .* ds

@dfunc logpdfUniform(a, b, x)   a   ( tmp = [a .<= x .<= b] .* (ds ./ (b - a)) ; isa(a, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfUniform(a, b, x)   b   ( tmp = [a .<= x .<= b] .* -(ds ./ (b - a)) ; isa(b, Real) ? sum(tmp) : tmp) .* ds
@dfunc logpdfUniform(a, b, x)   x   zero(x)

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


## Returns gradient expression of opex, with generic variables substituted with opex.args
function derive(opex::Expr, index::Integer, dsym::Union(Expr,Symbol))  # opex=:(-x);index=1;dsym=:y
	op = opex.args[1]  # operator
	vs = opex.args[1+index]
	ds = symbol("$(DERIV_PREFIX)$dsym")
	args = opex.args[2:end]
	arity = length(opex.args)-1

	# strip out module name (for logpdf...) if present
	# if isa(op, Expr) && op.head==:(.) && op.args[1]==:SimpleMCMC 
	# 	op = op.args[2].args[1]
	# end

	if has(rules, (op, index, arity))   # is operator/position defined in rules ?
		ex, pars = rules[(op, index, arity)]
		smap = { pars[i] => args[i] for i in 1:length(pars)}
		smap[:ds] = ds
		dexp = substSymbols(ex, smap)
	else
		error("[derive] Doesn't know how to derive operator $op on parameter $vs")
	end


	return :($(symbol("$(DERIV_PREFIX)$vs")) += $dexp )
end
