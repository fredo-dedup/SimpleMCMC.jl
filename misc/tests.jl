

include("../src/SimpleMCMC.jl")

A = [1, 2, 3]
model = quote
    x::real(1,3)

    y = x * A
    y ~ Normal(0,1)
end

SimpleMCMC.generateModelFunction(model, 1.0, false, true)
SimpleMCMC.generateModelFunction(model, 1.0, true, true)


SimpleMCMC.generateModelFunction(model, 1., true, true) 
SimpleMCMC.generateModelFunction(model, 1., false, true) 
myf, n, p, init = SimpleMCMC.generateModelFunction(model, [1., 0.05, 1.], true, false) 
myf(init)

##############################################
logpdfNormal(a,b,c) = SimpleMCMC.logpdfNormal(a,b,c)


#########################

begin   # ornstein
    srand(1)
    duration = 1000  # 1000 time steps
    mu0 = 10.  # target value
    tau0 = 20  # convergence time
    sigma0 = 0.1  # noise term

    x = fill(NaN, duration)
    x[1] = 1.
    for i in 2:duration
    	x[i] = x[i-1]*exp(-1/tau0) + mu0*(1-exp(-1/tau0)) +  sigma0*randn()
    end
    
    # model definition (note : rescaling on tau and mu)
    model = quote
        mu::real
        itau::real
        sigma::real

        itau ~ Gamma(2, 0.1)
        sigma ~ Gamma(2, 1)
        mu ~ Gamma(2, 1)

        fac = exp(- itau)
        dummy = 13.2
        dummy2 = sum(x)
        resid = x[2:end] - x[1:end-1] * fac - 10 * mu * (1. - fac)
        resid ~ Normal(0, sigma)
    end
end

include("../src/SimpleMCMC.jl")

res = SimpleMCMC.simpleRWM(model, 1000, 100, [1., 0.1, 1.])
res = SimpleMCMC.simpleHMC(model, 500, 100, [1., 0.1, 1.], 5, 0.002)

model = :(mu::real; mu ~ Normal(0,1))
res = SimpleMCMC.simpleRWM(model, 10000, 100)
res = SimpleMCMC.simpleHMC(model, 1000, 100, [1.], 5, 0.002)


###################

begin # linear
    # simulate dataset
    srand(1)
    n = 1000
    nbeta = 10 # number of predictors, including intercept
    X = [ones(n) randn((n, nbeta-1))]
    beta0 = randn((nbeta,))
    Y = X * beta0 + randn((n,))

    # define model
    model = quote
        vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
        resid = Y - X * vars
        resid += 0.
        # resid[nbeta] = 0.
        resid ~ Normal(0, 1.0)  
    end
end

include("../src/SimpleMCMC.jl")

    m = SimpleMCMC.MCMCModel()
    init = {:vars=> 1.0}
    init = {:x=> 1.0}
    SimpleMCMC.setInit!(m, init)
    SimpleMCMC.parseModel!(m, model)
    SimpleMCMC.parseModel!(m, :(a=3+x;2a))
    SimpleMCMC.unfold!(m)
    m.exprs
    SimpleMCMC.uniqueVars!(m)
    m.exprs
    SimpleMCMC.categorizeVars!(m)

    m.finalacc
    m.varsset
    m.pardesc
    m.accanc



SimpleMCMC.generateModelFunction(:(a=3+x;2a), debug=true, x=2.0, g=3)


SimpleMCMC.generateModelFunction(model, debug=true, vars=1.0)
SimpleMCMC.generateModelFunction(model, debug=true, vars=[1.0, 2])
SimpleMCMC.generateModelFunction(model, debug=true, vars=[1.0 2; 3 4], X=12)

SimpleMCMC.generateModelFunction(model, gradient=true, debug=true, vars=ones(nbeta))
myf, dummy = SimpleMCMC.generateModelFunction(model, vars=ones(nbeta))
myf(zeros(nbeta))
myf, dummy = SimpleMCMC.generateModelFunction(model, gradient =true, vars=ones(nbeta))
myf(zeros(nbeta))

include("../src/SimpleMCMC.jl")

SimpleMCMC.simpleAGD(model, vars=ones(nbeta))
SimpleMCMC.simpleRWM(model, steps=1e4, burnin=1e2, vars=ones(nbeta))

SimpleMCMC.simpleHMC(model, steps=1e4, burnin=1e2, vars=ones(nbeta), x=1, y=5)
SimpleMCMC.simpleNUTS(model, steps=500, burnin=1e2, vars=ones(nbeta))


SimpleMCMC.generateModelFunction(model, debug=true, gradient=true, vars=[1.0, 2], g=45)
SimpleMCMC.generateModelFunction(model, debug=true, vars=[1.0 2; 3 4], X=12)


# run random walk metropolis (10000 steps, 5000 for burnin)
res = SimpleMCMC.simpleRWM(model, 1000)
res.acceptRate  # acceptance rate
[ sum(res.params[:vars],2)./res.samples beta0 ] # show calculated and original coefs side by side
res = simpleHMC(model, 1000, 2, 0.05)
res = simpleNUTS(model, 1000)

####################

begin # binomial
    # simulate dataset
    srand(1)
    n = 1000
    nbeta = 10 # number of predictors, including intercept
    X = [ones(n) randn((n, nbeta-1))]
    beta0 = randn((nbeta,))
    Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

    # define model
    model = quote
        # vars::real(nbeta)

        vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
        prob = 1 / (1. + exp(X * vars)) 
        Y ~ Bernoulli(prob)
    end
end

SimpleMCMC.generateModelFunction(model, gradient=true, debug=true, vars=ones(nbeta))
myf, dummy = SimpleMCMC.generateModelFunction(model, gradient=true, vars=ones(nbeta))
myf(zeros(nbeta))

SimpleMCMC.generateModelFunction(model, debug=true, vars=ones(nbeta))

SimpleMCMC.simpleAGD(model, vars=ones(nbeta))


res = simpleRWM(model, 1000, 100)

mean(res.loglik)
[ sum(res.params[:vars],2) / res.samples beta0] # calculated vs original coefs
res = simpleHMC(model, 1000, 100, 2, 0.1)
res = simpleNUTS(model, 1000, 100)

##################################



function myf(a; b=false, c=true, ka...)
    println("a : $a, b: $b, c: $c, ka : $ka")
    for p in ka
        dump(p)
    end
end

myf(12,3)
myf(12, 3, "stop")
myf(12, c=3, b="stop")
myf(12, d=3, b="stop")

myf(12, b=3)
myf(12)

ndims([1. 2 ; 3 4])

using SimpleMCMC

m = SimpleMCMC.MCMCModel()
init = {:b=>123, :c=>[1,2], :d=>[1. 2 ; 3 4]}


function setInit!(m::MCMCModel, init)

    assert(length(init)>=1, "There should be at leat one parameter specified, none found")
    for p in init  # p = collect(init)[1]
        par = p[1]  # param symbol defined here
        def = p[2]

        assert(typeof(par) == Symbol, "[setInit] not a symbol in init param : $(par)")

        if isa(def, Real)  #  single param declaration
            push!(m.pars, MCMCParams(par, Integer[], m.bsize+1)) 
            m.bsize += 1
            push!(m.init, def)

        elseif isa(def, Array) && ndims(def) == 1
            nb = size(def,1)
            push!(m.pars, MCMCParams(par, Integer[nb], (m.bsize+1):(m.bsize+nb)))
            m.bsize += nb
            m.init = [m.init, def...]

        elseif isa(def, Array) && ndims(def) == 2
            nb1, nb2 = size(def)
            push!(m.pars, SimpleMCMC.MCMCParams(par, Integer[nb1, nb2], (m.bsize+1):(m.bsize+nb1*nb2))) 
            m.bsize += nb1*nb2
            m.init = [m.init, vec(def)...]

        else
            error("[setInit] forbidden parameter type for $(par)")
        end
    end

end
