
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

export x

res = SimpleMCMC.simpleRWM(model, 1000)
res = SimpleMCMC.simpleRWM(:(mu::real; mu ~ Normal(0,1)), 1000)


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
m.accanc & m.pardesc - Set(m.finalacc)
SimpleMCMC.backwardSweep!(m)
m.dexprs

begin
    body = SimpleMCMC.betaAssign(m)
    push!(body, :($ACC_SYM = 0.)) # acc init
    body = vcat(body, m.exprs)

    push!(body, :($(symbol("$DERIV_PREFIX$(m.finalacc)")) = 1.0))

    avars = m.accanc & m.pardesc - Set(m.finalacc) # remove accumulator, treated above  
    for v in avars 
        push!(body, :($(symbol("$DERIV_PREFIX$v")) = zero($(symbol("$v")))))
    end

    body = vcat(body, m.dexprs)

    if length(m.pars) == 1
        dn = symbol("$DERIV_PREFIX$(m.pars[1].sym)")
        dexp = :(vec([$dn]))  # reshape to transform potential matrices into vectors
    else
        dexp = {:vcat}
        # dexp = vcat(dexp, { (dn = symbol("$DERIV_PREFIX$(p.sym)"); :(vec([$DERIV_PREFIX$(p.sym)])) for p in pmap})
        dexp = vcat(dexp, { :( vec([$(symbol("$DERIV_PREFIX$(p.sym)"))]) ) for p in m.pars})
        dexp = expr(:call, dexp)
    end

    push!(body, :(($(m.finalacc), $dexp)))

    finalex = SimpleMCMC.tryAndFunc(body, true)
end

SimpleMCMC.logpdfNormal(0,1,1)
expr
template = :(module llmod; function ll(x); x;end; end)
template.args[3].args[4] = finalex
eval(template)
llmod.ll([1.,0.1,1.])

template = :(module llmod; using Main; function ll(x); x;end; end)
template.args[3].args[5] = :(function ll(x); Main.SimpleMCMC.logpdfNormal(0,1,x);end)
eval(template)
llmod.ll(1.)

dump(finalex)
dump(template.args[3].args[4])
template


eval(:(function test(y);y;end))
test("abcd")







dump(template.args[3].args[4].args[2].args[2])
    whos()

    myf = :(function gr(x); Main.SimpleMCMC.logpdfGamma(2,1,x);end)

    eval(:(module llmod; $myf; end) )
llmod.ll(10)
llmod.gr(0.9)

    eval(:(module loglik; function test(x); x+1;end; end) )
    eval(:(module $LLFUNC_SYM; function test(x); x+1;end; end) )
    @eval module loglik ; function test(x); x+5;end ; end
    test.loglik([1.,1.,1.])
    typeof(loglik.loglik)
    loglik.test
    test.loglik
    g = loglik.test
    g(5)

    dump(:(module loglik; function test(x); x+1;end; end) ,10)


    modexpr = expr(:module, true, LLFUNC_SYM, 
                   expr(:block, finalex))
    eval(modexpr)

    func = Main.eval(SimpleMCMC.tryAndFunc(body, true))





ll_func, nparams, pmap = SimpleMCMC.buildFunction(model)

ll_func(ones(10))

ll_func, nparams, pmap = SimpleMCMC.buildFunctionWithGradient(model)
ll_func(ones(10))

m = SimpleMCMC.parseModel(model)
[SimpleMCMC.betaAssign(m),             [:(__acc = 0.)],             m.source,             [:(return(__acc))] ]

SimpleMCMC.simpleRWM(model,1000,10)


m.dexprs
####

    allvarsset = mapreduce(p->SimpleMCMC.getSymbols(p.args[1]), union, exparray)

    avars = Set{Symbol}([p.sym for p in pmap]...)
    for ex2 in exparray # ex2 = exparray[1]
        lhs = SimpleMCMC.getSymbols(ex2.args[1])
        rhs = SimpleMCMC.getSymbols(ex2.args[2])

        length(intersect(rhs, avars)) > 0 ? avars = union(avars, lhs) : nothing
    end
    avars

    parents = Set{Symbol}(finalacc)
    for ex2 in reverse(exparray) # ex2 = reverse(exparray)[1]
        lhs = SimpleMCMC.getSymbols(ex2.args[1])
        rhs = SimpleMCMC.getSymbols(ex2.args[2])

        # println("$ex2 : $lhs = $rhs : $(intersect(lhs, parents)) => $(union(parents, rhs))....")
        length(lhs & parents) > 0 ? parents = parents | rhs : nothing
    end
    parents




length(parents)
length(avars)

for p in parents ; println(p, " ", has(avars, p)) ; end
for p in avars ; println(p, " ", has(parents, p)) ; end
for p in allvarsset ; println(p, " ", has(parents, p)) ; end

allvarsset - avars
allvarsset - parents
avars - parents
parents - avars


# julia> zb1 = zb0 + grady / (theta0 * L0)
# 3-element Float64 Array:
#  0.999419 
#  0.0248938
#  4.1756e-9


