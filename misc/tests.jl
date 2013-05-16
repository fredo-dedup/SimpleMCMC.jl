
begin
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

####
include("../src/SimpleMCMC.jl")

m = SimpleMCMC.parseModel(model)
SimpleMCMC.unfold!(m)
m.exprs
SimpleMCMC.uniqueVars!(m)
m.exprs
m.finalacc
SimpleMCMC.categorizeVars!(m)
m.varsset
m.accanc
m.pardesc

m.accanc - m.varsset - Set(SimpleMCMC.ACC_SYM, [p.sym for p in m.pars]...)
filter(e->contains([:(:), :end],e)))

SimpleMCMC.getSymbols(expr(:block, m.exprs...)) - Set(:(:), symbol("end"))

m.accanc & m.pardesc - Set(m.finalacc)
SimpleMCMC.backwardSweep!(m)
m.dexprs



allvarsset - avars
allvarsset - parents
avars - parents
parents - avars

SimpleMCMC.getSymbols(:a)
SimpleMCMC.getSymbols(:(a))
SimpleMCMC.getSymbols(:(a=2+b[c]))
SimpleMCMC.getSymbols(:(a=sin(2+b[c])))
SimpleMCMC.getSymbols(:(a[6]=sin(2+b[c])))
SimpleMCMC.getSymbols(:(a[z]=sin(2+b[c])))
SimpleMCMC.getSymbols(:(a[z]))


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
        vars::real(nbeta)

        vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
        resid = Y - X * vars
        resid += 0.
        resid[nbeta] = 0.
        resid ~ Normal(0, 1.0)  
    end
end

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


ma = { :(a), :(b=3), :(c=a+5)}
for e in ma
for i in 1:3
    ex2 = ma[i]
    ex2 = expr(:(=), :a, :c)
end
ma


#######################################
include("../src/SimpleMCMC.jl")

function test(el::Vector{Expr})
    subst = Dict{Symbol, Symbol}()
    used = Set(SimpleMCMC.ACC_SYM)

    for idx in 1:length(el) # idx=4
        ex2 = el[idx]

        # first, substitute variables names that have been renamed
        el[idx].args[2] = SimpleMCMC.substSymbols(el[idx].args[2], subst)

        # second, rename lhs symbol if set before
        lhs = collect(SimpleMCMC.getSymbols(ex2.args[1]))[1]  # there should be only one
        if contains(used, lhs) # if var already set once => create a new one
            subst[lhs] = gensym("$lhs") # generate new name, add it to substitution list for following statements
            el[idx].args[1] = SimpleMCMC.substSymbols(el[idx].args[1], subst)
        else # var set for the first time
            used |= Set(lhs) 
        end

    end

    el
    # (el, m.finalacc = has(subst, SimpleMCMC.ACC_SYM) ? subst[SimpleMCMC.ACC_SYM] : SimpleMCMC.ACC_SYM ) # keep reference of potentially renamed accumulator
end

el = [ :(a=2), :(b=a), :(c=z), :(a=c+5), :(x=2a), :(a = a +1), :(b=a)]
test(el)

