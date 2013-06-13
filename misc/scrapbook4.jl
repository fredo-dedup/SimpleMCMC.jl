## Normal distribution


d( log(1/sqrt(2sigma^2*pi)*exp(-(mu-x)^2/(2sigma^2))))/dx   dmu dsigma

@dfunc logpdfNormal(mu::Real, sigma, x)    mu     sum((x - mu) ./ (sigma .* sigma)) * ds
@dfunc logpdfNormal(mu::Array, sigma, x)   mu     (x - mu) ./ (sigma .* sigma) * ds
@dfunc logpdfNormal(mu, sigma::Real, x)    sigma  sum(((x - mu).*(x - mu) ./ (sigma.*sigma) - 1.) ./ sigma) * ds
@dfunc logpdfNormal(mu, sigma::Array, x)   sigma  ((x - mu).*(x - mu) ./ (sigma.*sigma) - 1.) ./ sigma * ds
@dfunc logpdfNormal(mu, sigma, x::Real)    x      sum((mu - x) ./ (sigma .* sigma)) * ds
@dfunc logpdfNormal(mu, sigma, x::Array)   x      (mu - x) ./ (sigma .* sigma) * ds

d( log(  1/2 * [ 1+ erf((x-mu)/(sqrt(2)*sigma))]) )/dx

@dfunc logcdfNormal(mu, sigma, x::Array)   x      exp( logpdfNormal(mu, sigma, x) ) ./ logcdfNormal(mu, sigma, x) * ds

## Uniform distribution
@dfunc logpdfUniform(a::Real, b, x)      a   sum((a .<= x .<= b) ./ (b - a)) * ds
@dfunc logpdfUniform(a::Array, b, x)     a   ((a .<= x .<= b) ./ (b - a)) * ds
@dfunc logpdfUniform(a, b::Real, x)      b   sum((a .<= x .<= b) ./ (a - b)) * ds
@dfunc logpdfUniform(a, b::Array, x)     b   ((a .<= x .<= b) ./ (a - b)) * ds
@dfunc logpdfUniform(a, b, x)            x   zero(x)

## Weibull distribution
@dfunc logpdfWeibull(sh::Real, sc, x)    sh  (r = x./sc ; sum(((1. - r.^sh) .* log(r) + 1./sh)) * ds)
@dfunc logpdfWeibull(sh::Array, sc, x)   sh  (r = x./sc ; ((1. - r.^sh) .* log(r) + 1./sh) * ds)
@dfunc logpdfWeibull(sh, sc::Real, x)    sc  sum(((x./sc).^sh - 1.) .* sh./sc) * ds
@dfunc logpdfWeibull(sh, sc::Array, x)   sc  ((x./sc).^sh - 1.) .* sh./sc * ds
@dfunc logpdfWeibull(sh, sc, x::Real)    x   sum(((1. - (x./sc).^sh) .* sh - 1.) ./ x) * ds
@dfunc logpdfWeibull(sh, sc, x::Array)   x   ((1. - (x./sc).^sh) .* sh - 1.) ./ x * ds

## Beta distribution
@dfunc logpdfBeta(a, b, x::Real)      x     sum((a-1) ./ x - (b-1) ./ (1-x)) * ds
@dfunc logpdfBeta(a, b, x::Array)     x     ((a-1) ./ x - (b-1) ./ (1-x)) * ds
@dfunc logpdfBeta(a::Real, b, x)      a     sum(digamma(a+b) - digamma(a) + log(x)) * ds
@dfunc logpdfBeta(a::Array, b, x)     a     (digamma(a+b) - digamma(a) + log(x)) * ds
@dfunc logpdfBeta(a, b::Real, x)      b     sum(digamma(a+b) - digamma(b) + log(1-x)) * ds
@dfunc logpdfBeta(a, b::Array, x)     b     (digamma(a+b) - digamma(b) + log(1-x)) * ds

## TDist distribution
@dfunc logpdfTDist(df, x::Real)     x     sum(-(df+1).*x ./ (df+x.*x)) .* ds
@dfunc logpdfTDist(df, x::Array)    x     (-(df+1).*x ./ (df+x.*x)) .* ds
@dfunc logpdfTDist(df::Real, x)     df    (tmp2 = (x.*x + df) ; sum( (x.*x-1)./tmp2 + log(df./tmp2) + digamma((df+1)/2) - digamma(df/2) ) / 2 .* ds )
@dfunc logpdfTDist(df::Array, x)    df    (tmp2 = (x.*x + df) ; ( (x.*x-1)./tmp2 + log(df./tmp2) + digamma((df+1)/2) - digamma(df/2) ) / 2 .* ds )

## Exponential distribution
@dfunc logpdfExponential(sc, x::Real)    x   sum(-1/sc) .* ds
@dfunc logpdfExponential(sc, x::Array)   x   (- ds ./ sc)
@dfunc logpdfExponential(sc::Real, x)    sc  sum((x-sc)./(sc.*sc)) .* ds
@dfunc logpdfExponential(sc::Array, x)   sc  (x-sc) ./ (sc.*sc) .* ds

## Gamma distribution
@dfunc logpdfGamma(sh, sc, x::Real)    x   sum(-( sc + x - sh.*sc)./(sc.*x)) .* ds
@dfunc logpdfGamma(sh, sc, x::Array)   x   (-( sc + x - sh.*sc)./(sc.*x)) .* ds
@dfunc logpdfGamma(sh::Real, sc, x)    sh  sum(log(x) - log(sc) - digamma(sh)) .* ds
@dfunc logpdfGamma(sh::Array, sc, x)   sh  (log(x) - log(sc) - digamma(sh)) .* ds
@dfunc logpdfGamma(sh, sc::Real, x)    sc  sum((x - sc.*sh) ./ (sc.*sc)) .* ds
@dfunc logpdfGamma(sh, sc::Array, x)   sc  ((x - sc.*sh) ./ (sc.*sc)) .* ds

## Cauchy distribution
@dfunc logpdfCauchy(mu, sc, x::Real)    x   sum(2(mu-x) ./ (sc.*sc + (x-mu).*(x-mu))) .* ds
@dfunc logpdfCauchy(mu, sc, x::Array)   x   (2(mu-x) ./ (sc.*sc + (x-mu).*(x-mu))) .* ds
@dfunc logpdfCauchy(mu::Real, sc, x)    mu  sum(2(x-mu) ./ (sc.*sc + (x-mu).*(x-mu))) .* ds
@dfunc logpdfCauchy(mu::Array, sc, x)   mu  (2(x-mu) ./ (sc.*sc + (x-mu).*(x-mu))) .* ds
@dfunc logpdfCauchy(mu, sc::Real, x)    sc  sum(((x-mu).*(x-mu) - sc.*sc) ./ (sc.*(sc.*sc + (x-mu).*(x-mu)))) .* ds
@dfunc logpdfCauchy(mu, sc::Array, x)   sc  (((x-mu).*(x-mu) - sc.*sc) ./ (sc.*(sc.*sc + (x-mu).*(x-mu)))) .* ds

## Log-normal distribution
@dfunc logpdflogNormal(lmu, lsc, x::Real)   x    ( tmp2=lsc.*lsc ; sum( (lmu - tmp2 - log(x)) ./ (tmp2.*x) ) .* ds )
@dfunc logpdflogNormal(lmu, lsc, x::Array)  x    ( tmp2=lsc.*lsc ; ( (lmu - tmp2 - log(x)) ./ (tmp2.*x) ) .* ds )
@dfunc logpdflogNormal(lmu::Real, lsc, x)   lmu  sum((log(x) - lmu) ./ (lsc .* lsc)) .* ds
@dfunc logpdflogNormal(lmu::Array, lsc, x)  lmu  ((log(x) - lmu) ./ (lsc .* lsc)) .* ds
@dfunc logpdflogNormal(lmu, lsc::Real, x)   lsc  ( tmp2=lsc.*lsc ; sum( (lmu.*lmu - tmp2 - log(x).*(2lmu-log(x))) ./ (lsc.*tmp2) ) .* ds )
@dfunc logpdflogNormal(lmu, lsc::Array, x)  lsc  ( tmp2=lsc.*lsc ; ( (lmu.*lmu - tmp2 - log(x).*(2lmu-log(x))) ./ (lsc.*tmp2) ) .* ds )

# TODO : find a way to implement multi variate distribs that goes along well with vectorization (Dirichlet, Categorical)
# TODO : other continuous distribs ? : Pareto, Rayleigh, Logistic, Levy, Laplace, Dirichlet, FDist
# TODO : other discrete distribs ? : NegativeBinomial, DiscreteUniform, HyperGeometric, Geometric, Categorical

## Bernoulli distribution (Note : no derivation on x parameter as it is an integer)
@dfunc logpdfBernoulli(p::Real, x)     p     sum(1. ./ (p - (1. - x))) * ds
@dfunc logpdfBernoulli(p::Array, x)    p     (1. ./ (p - (1. - x))) * ds

## Binomial distribution (Note : no derivation on x and n parameters as they are integers)
@dfunc logpdfBinomial(n, p::Real, x)   p    sum(x ./ p - (n-x) ./ (1 - p)) * ds
@dfunc logpdfBinomial(n, p::Array, x)  p    (x ./ p - (n-x) ./ (1 - p)) * ds

## Poisson distribution (Note : no derivation on x parameter as it is an integer)
@dfunc logpdfPoisson(lambda::Real, x)   lambda   sum(x ./ lambda - 1) * ds
@dfunc logpdfPoisson(lambda::Array, x)  lambda   (x ./ lambda - 1) * ds




