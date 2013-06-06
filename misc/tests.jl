

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

    ## process model parameters, rewrites ~ , ..
    m = SimpleMCMC.parseModel(model)

    ## checks initial values
    SimpleMCMC.setInit!(m, 1.0)

    ## process model expression
    SimpleMCMC.unfold!(m)
    SimpleMCMC.uniqueVars!(m)
    SimpleMCMC.categorizeVars!(m)

    ## build function expression
    body = SimpleMCMC.betaAssign(m)  # assigments beta vector -> model parameter vars
    push!(body, :($(SimpleMCMC.ACC_SYM) = 0.)) # initialize accumulator
    
    if gradient  # case with gradient


        body = Expr[ SimpleMCMC.betaAssign(m)..., 
                     :($(SimpleMCMC.ACC_SYM) = 0.), 
                     :(global vdict), 
                     :( vdict=Dict()),
                     m.exprs...,
                     :( vdict[$(expr(:quote, SimpleMCMC.ACC_SYM))] = $(SimpleMCMC.ACC_SYM) ),
                     [ :(vdict[$(expr(:quote, v))] = $v) for v in m.varsset]...]

    ev = m.accanc - m.varsset - Set(ACC_SYM, [p.sym for p in m.pars]...) # vars that are external to the model
    vhooks = expr(:block, [expr(:(=), v, expr(:., :Main, expr(:quote, v))) for v in ev]...) # assigment block

        body = :(let __beta = m.init; $vhooks; $(expr(:block, body...)); end)
        eval(body)
        vdict


        ###

        SimpleMCMC.backwardSweep!(m)

        body = vcat(body, m.exprs)
        push!(body, :($(symbol("$(SimpleMCMC.DERIV_PREFIX)$(m.finalacc)")) = 1.0))

        avars = m.accanc & m.pardesc - Set(m.finalacc) # remove accumulator, treated above  
        for v in avars 
            push!(body, :($(symbol("$(SimpleMCMC.DERIV_PREFIX)$v")) = zero($(symbol("$v")))))
        end
        body = vcat(body, m.dexprs)

        if length(m.pars) == 1
            dn = symbol("$(SimpleMCMC.DERIV_PREFIX)$(m.pars[1].sym)")
            dexp = :(vec([$dn]))  # reshape to transform potential matrices into vectors
        else
            dexp = {:vcat}
            dexp = vcat(dexp, { :( vec([$(symbol("$(SimpleMCMC.DERIV_PREFIX)$(p.sym)"))]) ) for p in m.pars})
            dexp = expr(:call, dexp)
        end

        push!(body, :(($(m.finalacc), $dexp)))

        # enclose in a try block
        body = expr(:try, expr(:block, body...),
                          :e, 
                          expr(:block, :(if e == "give up eval"; return(-Inf, zero($(SimpleMCMC.PARAM_SYM))); else; throw(e); end)))

    else  # case without gradient
        body = vcat(body, m.source.args)
        body = vcat(body, :(return($ACC_SYM)) )

        # enclose in a try block
        body = expr(:try, expr(:block, body...),
                          :e, 
                          expr(:block, :(if e == "give up eval"; return(-Inf); else; throw(e); end)))

    end

    # identify external vars and add definitions x = Main.x
    ev = m.accanc - m.varsset - Set(ACC_SYM, [p.sym for p in m.pars]...) # vars that are external to the model
    vhooks = expr(:block, [expr(:(=), v, expr(:., :Main, expr(:quote, v))) for v in ev]...) # assigment block

    # build and evaluate the let block containing the function and external vars hooks
    fn = gensym()
    body = expr(:function, expr(:call, fn, :($PARAM_SYM::Vector{Float64})), expr(:block, body) )
    body = :(let; global $fn; $vhooks; $body; end)



###############################################

let
    global vdict

    local x

    vdict = Dict()

    x = 12

    vdict[:x] = x
end

x
vdict[:x]
x = 34
vdict[:x]

    probe(s::Symbol) = eval(x)



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


##################

begin  #hierarchical
    ## generate data set
    N = 50 # number of observations
    D = 4  # number of groups
    L = 5  # number of predictors

    srand(1)
    mu0 = randn(1,L)
    sigma0 = rand(1,L)
    # model matrix nb group rows x nb predictors columns
    beta0 = Float64[randn()*sigma0[j]+mu0[j] for i in 1:D, j in 1:L] 

    oneD = ones(D)
    oneL = ones(L)

    ll = rand(1:D, N)  # mapping obs -> group
    X = randn(N, L)  # predictors
    Y = [rand(N) .< ( 1 ./ (1. + exp(- (beta0[ll,:] .* X) * oneL )))]

    ## define model
    model = quote
        mu::real(1,L)
        sigma::real(1,L)
        beta::real(D,L)

        mu ~ Normal(0, 1)
        sigma ~ Weibull(2, 1)

        beta ~ Normal(oneD * mu, oneD * sigma)

        effect = (beta[ll,:] .* X) * oneL
        prob = 1. / ( 1. + exp(- effect) )
        Y ~ Bernoulli(prob)
    end
end

# run random walk metropolis (10000 steps, 1000 for burnin)
res = simpleRWM(model, 1000, 100)

sum(res.params[:mu],3) / res.samples  # mu samples mean
sum(res.params[:sigma],3) / res.samples # sigma samples mean
sum(res.params[:beta],3) / res.samples # beta samples mean

# # run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 10 inner steps, 0.03 inner step size)
res = simpleHMC(model, 10000, 1000, 10, 0.03)


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
        resid[nbeta] = 0.
        resid ~ Normal(0, 1.0)  
    end
end

include("../src/SimpleMCMC.jl")

m = SimpleMCMC.MCMCModel()
init = {:vars=> 1.0}
SimpleMCMC.setInit!(m, init)
   m

SimpleMCMC.parseModel!(m, model)
typeof(m.source)
m.source.head
m.source.args

 ## process model
    SimpleMCMC.unfold!(m)
    m.exprs
    SimpleMCMC.uniqueVars!(m)
    m.exprs
    SimpleMCMC.categorizeVars!(m)

SimpleMCMC.generateModelFunction(model, debug=true, vars=1.0)
SimpleMCMC.generateModelFunction(model, debug=true, vars=[1.0, 2])
SimpleMCMC.generateModelFunction(model, debug=true, vars=[1.0 2; 3 4], X=12)

SimpleMCMC.generateModelFunction(model, gradient=true, debug=true, vars=ones(nbeta))
SimpleMCMC.generateModelFunction(model, debug=true, vars=[1.0, 2])
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
        vars::real(nbeta)

        vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
        prob = 1 / (1. + exp(X * vars)) 
        Y ~ Bernoulli(prob)
    end
end

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
