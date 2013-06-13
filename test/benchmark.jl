########################################################################
#    Benchmarking script for the log-likelihood function
#     and the samplers
########################################################################
# FIXME : this script now hangs julia

using SimpleMCMC

cd(joinpath(dirname(Base.find_in_path("SimpleMCMC")), "../test"))

BENCHFILE = "benchmarks.txt"
LIBVERSION = try ; readchomp(`git rev-parse --verify HEAD`)[1:6]; catch e; "-none-";end
JULVERSION = VERSION
# TODO : improve (gethostname hangs on my windows machine)
MACHINEID = string(hash(string(OS_NAME == :Windows ?  "-none-" : gethostname(), getipaddr())))[1:6]

fb = open(BENCHFILE, "a+")

## takes the best of 5 runs (adapted from examples/shootout macro)
macro timeit(ex, nit, name)
    @gensym t i
    quote
        $t = Inf
        for $i=1:5
            $t = min($t, @elapsed for j in 1:$nit; $ex; end)
        end
        println(fb, join([$(expr(:quote, name)), iround(time()), JULVERSION, 
            LIBVERSION, MACHINEID, round($t/$nit,9)], "\t"))
        println(join([$(expr(:quote, name)), iround(time()), JULVERSION, 
            LIBVERSION, MACHINEID, round($t/$nit,9)], "\t"))
    end
end


############  binomial reg test on 1000 obs x 10 predictors  ###############
# simulate dataset
srand(1)
n = 1000
nbeta = 10 # number of predictors, including intercept
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

# define model
model = quote
	vars ~ Normal(0, 1.0) 
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

init = {:vars => ones(nbeta)}

ll_func, nparams, pmap, init = generateModelFunction(model, gradient=true; init...) 
@timeit ll_func(init) 1000 binomial_function_with_gradient
ll_func, nparams, pmap, init = generateModelFunction(model; init...) 
@timeit ll_func(init) 1000 binomial_function_without_gradient

@timeit simpleRWM(model, steps=1000, burnin=100; init...) 1 binomial_RWM
@timeit simpleHMC(model, steps=1000, burnin=100, isteps=2, stepsize=0.1; init...) 1 binomial_HMC
@timeit simpleNUTS(model, steps=1000, burnin=100; init...) 1 binomial_NUTS


############  hierarchical reg test on 50 obs x 5 predictors  ###############
## generate data set
N = 50 # number of observations
D = 4  # number of groups
L = 5  # number of predictors

srand(1)
mu0 = randn(1,L)
sigma0 = rand(1,L)
beta0 = Float64[randn()*sigma0[j]+mu0[j] for i in 1:D, j in 1:L] 

oneD = ones(D)
oneL = ones(L)

ll = rand(1:D, N)  # mapping obs -> group
X = randn(N, L)  # predictors
Y = [rand(N) .< ( 1 ./ (1. + exp(- (beta0[ll,:] .* X) * oneL )))]

## define model
model = quote
    mu ~ Normal(0, 1)
    sigma ~ Weibull(2, 1)

    beta ~ Normal(oneD * mu, oneD * sigma)

    effect = (beta[ll,:] .* X) * oneL
    prob = 1. / ( 1. + exp(- effect) )
    Y ~ Bernoulli(prob)
end

init = {:mu=> ones(1,L),
        :sigma=> ones(1,L),
        :beta=> ones(D,L)}

ll_func, nparams, pmap, init = generateModelFunction(model, gradient=true; init...) 
@timeit ll_func(init) 1000 hierarchical_function_with_gradient
ll_func, nparams, pmap, init = generateModelFunction(model; init...) 
@timeit ll_func(init) 1000 hierarchical_function_without_gradient

@timeit simpleRWM(model, steps=1000, burnin=100;init...) 1 hierarchical_RWM
@timeit simpleHMC(model, steps=1000, burnin=100, isteps=10, stepsize=0.03; init...) 1 hierarchical_HMC
@timeit simpleNUTS(model, steps=10, burnin=5; init...) 1 hierarchical_NUTS   # poor perf of NUTS here, less iterations

############  Ornstein–Uhlenbeck process  ###############

srand(1)
duration = 1000  # 1000 time steps
mu0 = 10.  # target value
tau0 = 20  # convergence time
sigma0 = 0.1  # noise term

srand(1)
x = fill(NaN, duration)
x[1] = 1.
for i in 2:duration
    x[i] = x[i-1]*exp(-1/tau0) + mu0*(1-exp(-1/tau0)) +  sigma0*randn()
end

model = quote
    mu::real
    tau::real
    sigma::real

    tau ~ Uniform(0, 0.1)
    sigma ~ Uniform(0, 2)
    mu ~ Uniform(0, 2)

    fac = exp(- 0.001 / tau)
    resid = x[2:end] - x[1:end-1] * fac - 10 * mu * (1. - fac)
    resid ~ Normal(0, sigma)
end

init = {:mu=> 1.,
        :tau=> 0.05,
        :sigma=> 1.}

ll_func, nparams, pmap, init = generateModelFunction(model, gradient=true; init...) 
@timeit ll_func(init) 1000 Ornstein_function_with_gradient

ll_func, nparams, pmap, init = generateModelFunction(model; init...)
@timeit ll_func(init) 1000 Ornstein_function_without_gradient

@timeit simpleRWM(model, steps=1000, burnin=100; init...) 1 Ornstein_RWM
@timeit simpleHMC(model, steps=1000, burnin=100, isteps=5, stepsize=0.002; init...) 1 Ornstein_HMC
@timeit simpleNUTS(model, steps=1000, burnin=100; init...) 1 Ornstein_NUTS

############ close benchmark file ####################

close(fb)
