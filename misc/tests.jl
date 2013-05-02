using SimpleMCMC

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


####
    (model2, nparams, pmap) = SimpleMCMC.parseModel(model)
    exparray = SimpleMCMC.unfold(model2)
    exparray, finalacc = SimpleMCMC.uniqueVars(exparray)
    avars = SimpleMCMC.listVars(exparray, [p.sym for p in pmap])
    dmodel = backwardSweep(exparray, avars)

####

    allvarsset = mapreduce()

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
        length(intersect(lhs, parents)) > 0 ? parents = union(parents, rhs) : nothing
    end
    parents




length(parents)
length(avars)

for p in parents ; println(p, " ", has(avars, p)) ; end
for p in avars ; println(p, " ", has(parents, p)) ; end



end

# julia> zb1 = zb0 + grady / (theta0 * L0)
# 3-element Float64 Array:
#  0.999419 
#  0.0248938
#  4.1756e-9


