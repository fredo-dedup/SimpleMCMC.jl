########################################################################
#    Benchmarking script for the log-likelihood function
#     and the samplers
########################################################################

include("../src/SimpleMCMC.jl")

const BENCHFILE = "benchmarks.txt"
# const LIBVERSION = run(`git rev-parse --verify HEAD`)[1:6]
const LIBVERSION = "b79399"
const JULVERSION = VERSION

fb = open(BENCHFILE, "a+")


## takes the best of 5 runs (adapted from examples/shootout macro)
macro timeit(ex, nit)
    @gensym t i
    quote
        $t = Inf
        for $i=1:5
            $t = min($t, @elapsed for j in 1:$nit; $ex; end)
        end
        $t
    end
end

try
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
    	vars::real(nbeta)

    	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
    	prob = 1 / (1. + exp(X * vars)) 
    	Y ~ Bernoulli(prob)
    end

    ll_func, nparams, pmap = SimpleMCMC.buildFunctionWithGradient(model) # build function, count the number of parameters
    init = SimpleMCMC.setInit(1.0, nparams)

    t = @timeit ll_func(init) 1000
    println(fb, join(["binomial-function-with-gradient", JULVERSION, LIBVERSION, round(t,4)], "\t"))

    ll_func, nparams, pmap = SimpleMCMC.buildFunction(model) # build function, count the number of parameters
    init = SimpleMCMC.setInit(1.0, nparams)

    t = @timeit ll_func(init) 1000
    println(fb, join(["binomial-function-without-gradient", JULVERSION, LIBVERSION, round(t,4)], "\t"))

catch e
    close(fb)
end

close(fb)
