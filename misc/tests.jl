
include("../src/SimpleMCMC.jl")

A = [1, 2, 3]
model = quote
    x2::real(1,3)
    x1::real(2)
    x0::real

    y = x2 * A
    y ~ Normal(0,1)
end

SimpleMCMC.generateModelFunction(model, 1.0, false, true)
myf, dummy = SimpleMCMC.generateModelFunction(model, 1., false, false)
myf(ones(10)*0.5)
SimpleMCMC.generateModelFunction(model, 1.0, true, true)
myf, dummy = SimpleMCMC.generateModelFunction(model, 1., true, false)
myf(ones(10)*0.5)
myf(zeros(30)+0.1)

myf([1., 1., 1.])

SimpleMCMC.generateModelFunction(model, [1., 0.1, 1.], true, true)
myf, dummy = SimpleMCMC.generateModelFunction(model, [1., 0.1, 1.], true, false)
myf([1., 0.1, 1.])
res = simpleHMC(model, 1000, 100, [1., 0.1, 1.], 5, 0.002)


SimpleMCMC.generateModelFunction(expr(:block, :(x::real ; y = max(x,arg2) ; y ~ TestDiff())), [1.0], true, true)
SimpleMCMC.getSymbols(:((x.>arg2)))
SimpleMCMC.getSymbols(:([zz.>=x.>arg2]))
dump(:((x.>arg2)))
dump(:((z.>x.>arg2)))

#########################

begin # binomial
    srand(1)
    n = 1000
    nbeta = 10 # number of predictors, including intercept
    X = [ones(n) randn((n, nbeta-1))]
    beta0 = randn((nbeta,))
    Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

    # define model
    model = quote
        vars::real(nbeta)

        vars ~ Normal(0, 1.0) 
        prob = 1 / (1. + exp(X * vars)) 
        Y ~ Bernoulli(prob)
    end
end

    ll_func, nparams, pmap, init = SimpleMCMC.generateModelFunction(model, 1.0, true, false) 
    @timeit ll_func(init) 1000 binomial_function_with_gradient

    ll_func, nparams, pmap, init = SimpleMCMC.generateModelFunction(model, 1.0, false, false) 
    @timeit ll_func(init) 1000 binomial_function_without_gradient


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
        vars::real(nbeta)

        vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
        resid = Y - X * vars
        resid += 0.
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


let 
    global f57003
    local x = Main.x
    local tmp56963 = -(0.001)
    local tmp56965 = x[2:end]
    local tmp56966 = x[1:-(end,1)]
    local tmp56999 = -(tmp56963)
    function f57003(__beta::Vector{Float64})
        try 
            local mu = __beta[1]
            local tau = __beta[2]
            local sigma = __beta[3]
            local __acc = 0.0
            local d__acc56976 = 1.0
            local dtmp56969 = 0.0
            local dtmp56968 = zeros(Float64,(999,))
            local d__acc56973 = 0.0
            local dtmp56964 = 0.0
            local d__acc56975 = 0.0
            local dtmp56971 = 0.0
            local d__acc56974 = 0.0
            local dsigma = 0.0
            local dtmp56972 = 0.0
            local dfac = 0.0
            local dtau = 0.0
            local dmu = 0.0
            local dtmp56967 = zeros(Float64,(999,))
            local dtmp56962 = 0.0
            local dtmp56970 = 0.0
            local dresid = zeros(Float64,(999,))
            local dtmp56961 = 0.0
            local dtmp56960 = 0.0
            local tmp56960 = logpdfUniform(0,0.1,tau)
            local __acc56973 = +(__acc,tmp56960)
            local tmp56961 = logpdfUniform(0,2,sigma)
            local __acc56974 = +(__acc56973,tmp56961)
            local tmp56962 = logpdfUniform(0,2,mu)
            local __acc56975 = +(__acc56974,tmp56962)
            local tmp56964 = /(tmp56963,tau)
            local fac = exp(tmp56964)
            local tmp56967 = *(tmp56966,fac)
            local tmp56968 = -(tmp56965,tmp56967)
            local tmp56969 = -(1.0,fac)
            local tmp56970 = *(mu,tmp56969)
            local tmp56971 = *(10,tmp56970)
            local resid = -(tmp56968,tmp56971)
            local tmp56972 = logpdfNormal(0,sigma,resid)
            local __acc56976 = +(__acc56975,tmp56972)
            d__acc56975 = +(d__acc56975,sum(d__acc56976))
            dtmp56972 = +(dtmp56972,sum(d__acc56976))
            local tmp56980 = -(resid,0)
            local tmp56981 = .^(tmp56980,2)
            local tmp56982 = .^(sigma,2)
            local tmp56983 = ./(tmp56981,tmp56982)
            local tmp56984 = -(tmp56983,1.0)
            local tmp56985 = ./(tmp56984,sigma)
            local tmp56986 = *(tmp56985,dtmp56972)
            local tmp56987 = sum(tmp56986)
            dsigma = +(dsigma,.*(tmp56987,dtmp56972))
            local tmp56988 = -(0,resid)
            local tmp56989 = .^(sigma,2)
            local tmp56990 = ./(tmp56988,tmp56989)
            local tmp56991 = *(tmp56990,dtmp56972)
            dresid = +(dresid,.*(tmp56991,dtmp56972))
            dtmp56968 = +(dtmp56968,+(dresid))
            local tmp56992 = sum(dresid)
            dtmp56971 = +(dtmp56971,-(tmp56992))
            local tmp56993 = .*(dtmp56971,10)
            dtmp56970 = +(dtmp56970,sum(tmp56993))
            local tmp56994 = .*(dtmp56970,tmp56969)
            dmu = +(dmu,sum(tmp56994))
            local tmp56995 = .*(dtmp56970,mu)
            dtmp56969 = +(dtmp56969,sum(tmp56995))
            local tmp56996 = sum(dtmp56969)
            dfac = +(dfac,-(tmp56996))
            dtmp56967 = +(dtmp56967,-(dtmp56968))
            local tmp56997 = .*(dtmp56967,tmp56966)
            dfac = +(dfac,sum(tmp56997))
            local tmp56998 = exp(tmp56964)
            dtmp56964 = +(dtmp56964,.*(tmp56998,dfac))
            local tmp57000 = .*(tau,tau)
            local tmp57001 = ./(tmp56999,tmp57000)
            local tmp57002 = .*(tmp57001,dtmp56964)
            dtau = +(dtau,sum(tmp57002))
            d__acc56974 = +(d__acc56974,sum(d__acc56975))
            dtmp56962 = +(dtmp56962,sum(d__acc56975))
            dmu = +(dmu,zero(mu))
            d__acc56973 = +(d__acc56973,sum(d__acc56974))
            dtmp56961 = +(dtmp56961,sum(d__acc56974))
            dsigma = +(dsigma,zero(sigma))
            dtmp56960 = +(dtmp56960,sum(d__acc56973))
            dtau = +(dtau,zero(tau))
            (__acc56976,vcat(vec([dmu]),vec([dtau]),vec([dsigma])))
        catch e
            if (e=="give up eval") 
                return (-(Inf),zero(__beta))
            else   
                throw(e)
            end
        end
    end
end